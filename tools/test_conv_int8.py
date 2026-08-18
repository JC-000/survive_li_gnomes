#!/usr/bin/env python3
"""Prove the uint8-biased int8 convolution, and prove its int32 bound.

    python3 tools/test_conv_int8.py

This is the host reference for the fallback that `tools/cnn_probe.py` section 4
measures and that `src/` would carry if `emlearn_cnn_int8` turns out not to load
on this board. It exists for the same reason `tools/test_vad.py` exists: the
device version is `@micropython.viper`, viper's `int` is a 32-bit machine word
that **wraps silently**, and neither CPython nor MicroPython bytecode will
reproduce that, because both carry unbounded integers. A bound that is only
argued in a comment is a bound nobody has tested.

Two separate claims are checked here, and they fail differently.

## 1. The bias trick is exact, not approximate

A `ptr8` load in viper is zero-extended, exactly as a `ptr16` load is
(docs/speech.md, "What the viper port cost to get right"). Sign-extending every
activation and every weight would add statements to the innermost loop in the
whole program, and viper's cost is very nearly statement count. So both are
stored **biased by +128**, as uint8, and the inner loop is a plain unsigned
multiply-accumulate. The bias comes back out algebraically:

    sum (W-128)(X-128) = sum W*X  -  128*sum X  -  128*sum W  +  128*128*n

`sum W` and `128*128*n` depend only on the filter, so they fold into the bias
when the weights are packed. `sum X` depends on the input patch but not on the
output channel, so it is computed once per patch and reused across all `cho`.

That is exactly the manoeuvre the DTW inner loop already makes, where templates
are stored as uint16 biased by +32768 so `ptr16` can read them without sign
extension and the bias cancels in a difference. It is exact in integers, and
`test_equivalence` checks it is exact over random and adversarial inputs rather
than over one convenient example.

## 2. The requant multiply stays at half of int32

The accumulator is not the tight stage and it is worth being clear about that,
because the natural worry is the wrong one. Each term is `(W-128)(X-128)` with
both factors in `[-128, 127]`, so `|term| <= 16384` and
`|acc| <= 16384 * kh*kw*chi`. At TinyMaix's own widest, `kh*kw*chi = 2304`, that
is 37.7e6 -- **1.76% of int32**, 56 times clear. Nothing to defend.

The tight stage is the requantisation, where the accumulator meets a
per-channel multiplier. Taking the multiplier in Q15 gives a product of up to
`65535 * 32767 = 2.147e9`, **99.99% of int32**, which is not a margin. So:

- the accumulator is shifted right by `r` first, with `r` chosen from the
  algebraic bound so that `|acc >> r| < 65536`;
- the multiplier is **Q14**, at most 16383.

The product is then at most `65535 * 16383 = 1.0737e9` -- **50.0% of int32, a
full factor of two clear**. That is the same margin the FFT twiddle keeps, and
`docs/speech.md` reaches for the same remedy (Q14 instead of Q15) for the same
stage if the FFT's invariant ever changes. The cost is one bit of multiplier
precision, on a value whose result is then quantised to eight bits anyway.

`test_bound` drives the accumulator to its algebraic maximum -- every activation
and every weight at an extreme -- and asserts the shift keeps it inside 16 bits
and the product inside half of int32. Unlike the pre-emphasis case, where the
deliberately hostile signal turned out *milder* than real speech because
saturation clamped it first, here the worst case really is reachable: nothing
stands between the input and the accumulator.
"""

import random
import sys

FAILURES = []


def check(name, ok, detail=""):
    if ok:
        print("  ok    %s" % name)
    else:
        print("  FAIL  %s  %s" % (name, detail))
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# The reference: a direct signed int8 convolution, no tricks.
# ---------------------------------------------------------------------------

def conv_reference(src, ih, iw, chi, wts, cho, kh, kw, sh, sw, bias):
    """Valid padding, HWC layout, signed int8 values, int32 accumulator.

    `src` and `wts` hold true signed values in [-128, 127]. This is the thing
    the biased version has to agree with.
    """
    oh = (ih - kh) // sh + 1
    ow = (iw - kw) // sw + 1
    out = []
    for oy in range(oh):
        for ox in range(ow):
            for oc in range(cho):
                acc = bias[oc]
                for ky in range(kh):
                    for kx in range(kw):
                        for c in range(chi):
                            s = src[((oy * sh + ky) * iw + (ox * sw + kx)) * chi + c]
                            w = wts[((oc * kh + ky) * kw + kx) * chi + c]
                            acc += s * w
                out.append(acc)
    return out, oh, ow


# ---------------------------------------------------------------------------
# The port: what the viper function does, statement for statement.
# ---------------------------------------------------------------------------

def pack_weights(wts, cho, kh, kw, chi, bias):
    """Bias the weights to uint8 and fold the constant term into the bias.

    Returns (uint8 weights, adjusted int32 bias), where the adjustment is
    **`-128 * sum(W)` over the signed weights, and nothing else**.

    That is worth spelling out, because the expansion written in terms of the
    stored uint8 values looks like it needs a `+128*128*n` term as well:

        sum (Wu-128)(Xu-128) = sum Wu*Xu - 128*sum Xu - 128*sum Wu + 128*128*n

    and `sum Wu = sum W + 128*n`, so the `-128*sum Wu` already carries a
    `-128*128*n` that cancels the `+128*128*n` exactly. Adding both leaves the
    accumulator wrong by `128*128*n` -- 1179648 for a 3x24x1 filter, which is
    not a rounding error but does not crash either. The first version of this
    file had exactly that bug and `test_equivalence` is what found it.
    """
    n = kh * kw * chi
    packed = [w + 128 for w in wts]
    adjusted = []
    for oc in range(cho):
        sw = sum(wts[oc * n:(oc + 1) * n])
        adjusted.append(bias[oc] - 128 * sw)
    return packed, adjusted


def conv_biased(src_u8, ih, iw, chi, wts_u8, cho, kh, kw, sh, sw, bias_adj):
    """Plain-Python twin of `_conv2d_u8` in tools/cnn_probe.py.

    Returns (accumulators, peak |acc|). Deliberately mirrors the viper loop
    structure -- one contiguous run of `kw*chi` per patch row -- rather than
    being written the clearest way, because the point is to check *that* code.
    """
    oh = (ih - kh) // sh + 1
    ow = (iw - kw) // sw + 1
    kwc = kw * chi
    krow = iw * chi
    ksz = kh * kwc

    out = []
    peak = 0
    for oy in range(oh):
        for ox in range(ow):
            base = (oy * sh) * krow + (ox * sw) * chi

            psum = 0
            sp = base
            for _ in range(kh):
                for i in range(kwc):
                    psum += src_u8[sp + i]
                sp += krow
            psum = psum << 7          # the -128 * sum X term

            wp = 0
            for oc in range(cho):
                acc = 0
                sp = base
                wq = wp
                for _ in range(kh):
                    for i in range(kwc):
                        acc += src_u8[sp + i] * wts_u8[wq + i]
                    sp += krow
                    wq += kwc
                acc = acc - psum + bias_adj[oc]
                out.append(acc)
                if abs(acc) > peak:
                    peak = abs(acc)
                wp += ksz
    return out, peak


def requant(acc, mult_q14, shift, rsh):
    """The device's requantisation, in the order the viper code does it."""
    a = acc >> rsh                    # arithmetic; floors on both sides
    v = (a * mult_q14) >> shift
    v += 128
    if v < 0:
        v = 0
    if v > 255:
        v = 255
    return v


def choose_rshift(kh, kw, chi, max_bias=0):
    """Smallest r with |acc >> r| < 65536, from the algebra rather than a run."""
    bound = 16384 * kh * kw * chi + abs(max_bias)
    rsh = 0
    while (bound >> rsh) >= 65536:
        rsh += 1
    return rsh, bound


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

SHAPES = (
    # (ih, iw, chi, kh, kw, cho, sh, sw), with a comment on why each is here
    (64, 24, 1, 3, 24, 16, 1, 1),   # topology C layer 1: spans the whole mel axis
    (62, 1, 16, 3, 1, 24, 2, 1),    # topology C layer 2: 1-D in time, strided
    (30, 1, 24, 3, 1, 32, 2, 1),    # topology C layer 3
    (12, 12, 1, 3, 3, 8, 2, 2),     # the MNIST shape, strided both ways
    (5, 5, 3, 3, 3, 4, 1, 1),       # small and square, multi-channel
    (4, 4, 2, 4, 4, 3, 1, 1),       # kernel exactly fills the input: oh = ow = 1
    (7, 1, 5, 7, 1, 2, 1, 1),       # tall thin kernel, also exactly filling
    (9, 3, 2, 3, 3, 1, 3, 1),       # stride equal to the kernel: no overlap
)


def test_equivalence():
    """The biased path must equal the signed reference. Exactly, every time."""
    print("equivalence with a direct signed convolution")
    rng = random.Random(20260818)

    for shape in SHAPES:
        ih, iw, chi, kh, kw, cho, sh, sw = shape
        n = kh * kw * chi

        for trial, mode in enumerate(("random", "extremes", "all_min", "all_max")):
            if mode == "random":
                src = [rng.randint(-128, 127) for _ in range(ih * iw * chi)]
                wts = [rng.randint(-128, 127) for _ in range(cho * n)]
            elif mode == "extremes":
                # Only the endpoints, which is where an off-by-one in the bias
                # would show up and a uniform random draw would not.
                src = [rng.choice((-128, 127)) for _ in range(ih * iw * chi)]
                wts = [rng.choice((-128, 127)) for _ in range(cho * n)]
            elif mode == "all_min":
                src = [-128] * (ih * iw * chi)
                wts = [-128] * (cho * n)
            else:
                src = [127] * (ih * iw * chi)
                wts = [127] * (cho * n)

            bias = [rng.randint(-5000, 5000) for _ in range(cho)]

            want, oh, ow = conv_reference(src, ih, iw, chi, wts, cho, kh, kw, sh, sw, bias)
            packed_w, bias_adj = pack_weights(wts, cho, kh, kw, chi, bias)
            src_u8 = [s + 128 for s in src]
            got, _ = conv_biased(src_u8, ih, iw, chi, packed_w, cho, kh, kw, sh, sw, bias_adj)

            same = want == got
            first = ""
            if not same:
                for i, (a, b) in enumerate(zip(want, got)):
                    if a != b:
                        first = "first mismatch at %d: want %d got %d" % (i, a, b)
                        break
            check("%s %s -> %dx%dx%d" % (shape, mode, oh, ow, cho), same, first)


def test_bound():
    """Drive the accumulator to its algebraic maximum and check it stays put.

    Two things are asserted, and only the second is about overflow:

      - `|acc >> r| < 65536`, i.e. the pre-shift really does land the
        accumulator in 16 bits, so the requant multiply is bounded;
      - `|a| * 16383 <= 2^30`, i.e. 50% of int32.

    The maximum is reached by making every product `+16384`, which needs every
    activation at -128 and every weight at -128 (or both at their positive
    extremes, which reaches 16129 -- the negative pair is the true worst case
    and it is easy to miss).
    """
    print()
    print("int32 bound at the algebraic worst case")

    for shape in SHAPES:
        ih, iw, chi, kh, kw, cho, sh, sw = shape
        n = kh * kw * chi

        max_bias = 5000
        rsh, bound = choose_rshift(kh, kw, chi, max_bias)

        src = [-128] * (ih * iw * chi)
        wts = [-128] * (cho * n)
        bias = [max_bias] * cho

        packed_w, bias_adj = pack_weights(wts, cho, kh, kw, chi, bias)
        src_u8 = [s + 128 for s in src]
        _, peak = conv_biased(src_u8, ih, iw, chi, packed_w, cho, kh, kw, sh, sw, bias_adj)

        # The accumulator itself, before any shift.
        acc_pct = 100.0 * peak / 2147483648.0
        shifted = peak >> rsh
        product = shifted * 16383
        prod_pct = 100.0 * product / 2147483648.0

        check("%s acc %d = %.3f%% of int32, matches the algebra (%d)"
              % (shape, peak, acc_pct, bound),
              peak <= bound,
              "measured %d exceeds the proved %d" % (peak, bound))
        check("%s |acc >> %d| = %d < 65536" % (shape, rsh, shifted),
              shifted < 65536,
              "pre-shift left %d, the requant multiply would overflow" % shifted)
        check("%s requant product %d = %.2f%% of int32"
              % (shape, product, prod_pct),
              product <= (1 << 30),
              "%.2f%% -- above the 50%% the design claims" % prod_pct)


def test_q15_would_overflow():
    """The negative check, so the Q14 choice is evidenced rather than asserted.

    If Q15 were fine there would be no reason to give up the bit, and the next
    person to read the code would quite reasonably put it back. So: show that
    it is not fine.
    """
    print()
    print("why Q14 and not Q15")
    worst = 65535
    q15 = worst * 32767
    q14 = worst * 16383
    check("Q15 product %d is %.2f%% of int32 -- no margin"
          % (q15, 100.0 * q15 / 2147483648.0),
          q15 > (1 << 30),
          "expected Q15 to exceed half of int32")
    check("Q14 product %d is %.2f%% of int32 -- a factor of two clear"
          % (q14, 100.0 * q14 / 2147483648.0),
          q14 <= (1 << 30))


def test_requant_monotonic():
    """Requantisation must not fold, and the clamp must be a clamp.

    A sign error in the shift shows up as a non-monotonic mapping, which is the
    kind of fault that leaves a model working badly rather than not working.
    """
    print()
    print("requantisation is monotonic and clamps")
    rsh, mult, shift = 6, 12000, 14
    prev = None
    monotonic = True
    for acc in range(-4000000, 4000001, 9973):
        v = requant(acc, mult, shift, rsh)
        if prev is not None and v < prev:
            monotonic = False
            break
        prev = v
    check("monotonic over the accumulator range", monotonic)
    check("clamps low", requant(-1 << 29, mult, shift, rsh) == 0)
    check("clamps high", requant(1 << 29, mult, shift, rsh) == 255)


def test_arithmetic_shift():
    """`>>` must floor on negatives on both sides, as docs/speech.md requires.

    CPython floors, ARM's ASR floors, and viper's `int` is a signed machine
    word. `// 2` does not floor the same way for every case anyone reaches for
    as a substitute, so this pins the one that is used.
    """
    print()
    print("shifts floor on negatives")
    check("-3 >> 1 == -2", (-3 >> 1) == -2)
    check("-1 >> 4 == -1", (-1 >> 4) == -1)
    check("-65537 >> 1 == -32769", (-65537 >> 1) == -32769)


def main():
    test_equivalence()
    test_bound()
    test_q15_would_overflow()
    test_requant_monotonic()
    test_arithmetic_shift()

    print()
    if FAILURES:
        print("%d FAILED" % len(FAILURES))
        for name in FAILURES:
            print("  %s" % name)
        return 1
    print("all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
