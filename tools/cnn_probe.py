#!/usr/bin/env python3
"""Settle, in one board session, whether a trained CNN can run on this device.

    uvx mpremote connect /dev/cu.usbmodem101 cp <stage>/emlearn_cnn_int8.mpy :
    uvx mpremote connect /dev/cu.usbmodem101 cp -r <stage>/cnn :
    uvx mpremote connect /dev/cu.usbmodem101 run tools/cnn_probe.py

Runs on the device. Touches no peripheral -- no panel, no codec, no I2C -- so it
is safe to run against a board that is mid-way through anything else, and it
leaves nothing behind but the files copied above.

## Status: this file has never been run on hardware

It was written and dry-run against CPython with viper stubbed, which proves it
*executes* -- every branch, every format string -- and proves nothing else. The
expected values quoted in the comments below (~12 KB import cost, 2x the file
resident, cycles per MAC) are **predictions from reading source and from this
project's earlier viper measurements**, not observations. The whole point of the
script is that they cannot be settled any other way.

When it does run, the numbers it prints replace those predictions, and
`docs/cnn-on-device.md` should be updated from its output rather than from this
docstring.

## What it is for

The keyword spotter ships DTW over MFCCs, which needs templates enrolled through
this board's own microphone, because cross-microphone enrolment collapses top-1
from 69/70 to 36/70 (docs/speech.md). A trained classifier would not care whose
microphone it heard, which is worth having, but only if it can run here.

`emlearn_cnn_int8` is the only route that keeps a trained model inside
MicroPython: a native `.mpy` wrapping Sipeed's TinyMaix, int8 Conv2d / dwConv2d
/ FC / GAP / Softmax, loading a `.tmdl`. emlearn publishes a prebuilt binary for
`armv7emsp` at mpy 6.3, which is what this board reports -- but upstream states
it is tested on x64 and xtensawin only, so **a clean import here is unproven**,
and unproven is the whole reason this file exists.

Everything else in this experiment is downstream of the answer, so the answer is
wanted early and in one pass.

## Why the MNIST model and not ours

Sections 1 to 3 use emlearn's own shipped MNIST model and its ten labelled
example digits. That is deliberate: it is known-good upstream, so a wrong answer
here is the *port's* fault and nothing else's, and it needs nothing from the
model that is still being trained. Do not debug a native-code port and an
untrained network at the same time.

**A clean import is not the test.** A native module can load and then compute
nonsense if the code generation is subtly wrong for this core, and nothing about
that raises. So section 3 checks the ten digits come back correctly classified;
that is what actually proves the ARM path works.

## Staging the files

Fetched on the Mac -- this board is an RP2350A with no networking, so it cannot
`mip install` anything itself:

    curl -O https://emlearn.github.io/emlearn-micropython/builds/latest/armv7emsp_6.3/emlearn_cnn_int8.mpy
    B=https://raw.githubusercontent.com/emlearn/emlearn-micropython/master/examples/mnist_cnn
    curl -o cnn/mnist_cnn_int8.tmdl $B/mnist_cnn_int8.tmdl
    for i in 0 1 2 3 4 5 6 7 8 9; do curl -o cnn/mnist_$i.bin $B/data/mnist_example_$i.bin; done

Optionally, alongside them, the keyword model itself: `cnn/model.tmdl` plus any
number of `cnn/kw_<class>_<n>.bin` input patches, raw uint8 in the model's own
input shape, named for the class the host interpreter predicts on those exact
bytes. Section 3b picks them up if present and is skipped silently if not.
Run `python3 tools/tmdl_info.py model.tmdl` first -- it refuses the cases that
corrupt the heap instead of raising.

And optionally `cnn/overrun.tmdl`: a model that `tmdl_info.py` reports as
needing more activation buffer than its file length, copied across *without*
`--pad`. Section 6 uses it to confirm the overrun on hardware. It is opt-in
under its own filename because it deliberately corrupts memory; reset the board
afterwards.

(`mpremote mip install <url>` would also work -- mip resolves and downloads on
the *host* and copies the result over -- but a plain `cp` of a file already on
disk is one less thing to be wrong about at the bench.)

## Section 4 runs regardless

If the import fails, that is a real finding and not a dead end: the fallback is
hand-rolled int8 convolution in `@micropython.viper`. Section 4 measures that
fallback whether or not section 1 succeeded, because the two numbers are only
useful next to each other -- and because the cost of the fallback is what
decides how large a model the training side is allowed to build.
"""

import array
import gc
import os
import sys
import time

DEV = "/cnn"          # where the .tmdl and the digits were copied
CLOCK_HZ = 150_000_000


def heap():
    gc.collect()
    return gc.mem_free()


def biggest():
    """Largest contiguous allocation, found by bisection rather than assumed.

    An earlier probe in this project read the heap as too small to hold the
    capture buffer, and it was the probe's own fragmentation talking. So this
    collects first and reports both numbers.
    """
    gc.collect()
    lo, hi = 0, gc.mem_free()
    while lo < hi:
        mid = (lo + hi + 1) // 2
        try:
            buf = bytearray(mid)
            del buf
            lo = mid
        except MemoryError:
            hi = mid - 1
        gc.collect()
    return lo


def rule(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# 0. What this build actually is
# ---------------------------------------------------------------------------

ARCH = {
    0: "none", 1: "x86", 2: "x64", 3: "armv6", 4: "armv6m", 5: "armv7m",
    6: "armv7em", 7: "armv7emsp", 8: "armv7emdp", 9: "xtensa", 10: "xtensawin",
    11: "rv32imc",
}

rule("0. build")

print("MicroPython  %s" % ".".join(str(v) for v in sys.implementation.version))
mpy = getattr(sys.implementation, "_mpy", None)
if mpy is None:
    print("_mpy         absent -- this build cannot load native .mpy at all")
else:
    # Low byte is the .mpy version and sub-version; the rest names the native
    # architecture the loader will accept. A mismatch here is the whole
    # question, so read it off the device rather than trusting a note.
    ver, sub, arch = mpy & 0xFF, (mpy >> 8) & 0x3, mpy >> 10
    print("mpy version  %d.%d" % (ver, sub))
    print("native arch  %s (%d)" % (ARCH.get(arch, "unknown"), arch))
    print("             the .mpy staged for this run is armv7emsp / 6.3")

free0 = heap()
print("heap free    %d B" % free0)
print("largest      %d B" % biggest())


# ---------------------------------------------------------------------------
# 1. Does the native module import
# ---------------------------------------------------------------------------

rule("1. import emlearn_cnn_int8")

before = heap()
cnn_mod = None
import_error = None
try:
    import emlearn_cnn_int8 as cnn_mod
    after = heap()
    print("IMPORTED. heap cost %d B" % (before - after))
    print("  (expect ~12 KB: ~5.5 KB of code plus TinyMaix's static arrays --")
    print("   sbuf 2304 B, sumscale 4000 B, k_oft 100 B)")
except Exception as exc:      # noqa: BLE001 -- reporting it *is* the job
    import_error = exc
    print("FAILED: %s: %s" % (type(exc).__name__, exc))
    print()
    print("  This is a result, not an accident. It means the trained-model path")
    print("  must be the hand-rolled viper convolution measured in section 4,")
    print("  and the model has to be small enough for it. Report it upstream.")


# ---------------------------------------------------------------------------
# 2. Does it load a model
# ---------------------------------------------------------------------------

model = None
n_classes = 0

if cnn_mod is not None:
    rule("2. load the shipped MNIST model")
    try:
        with open(DEV + "/mnist_cnn_int8.tmdl", "rb") as handle:
            blob = handle.read()
        print("model file   %d B" % len(blob))

        before = heap()
        data = array.array("B", blob)
        del blob
        gc.collect()
        model = cnn_mod.new(data)
        peak_free = gc.mem_free()
        del data
        after = heap()

        print("LOADED.")
        print("  heap at rest %d B" % (before - after))
        print("               expect ~2x the file: emlearn keeps a copy of the")
        print("               model and an equal-sized activation scratch")
        print("  heap peak    %d B" % (before - peak_free))
        print("               the caller's array('B') is still alive at that point")
        dims = model.output_dimensions()
        n_classes = dims[0]
        print("  output dims  %s" % (dims,))
    except Exception as exc:      # noqa: BLE001
        print("FAILED: %s: %s" % (type(exc).__name__, exc))
        model = None


# ---------------------------------------------------------------------------
# 3. Does it compute the right answers, and how fast
# ---------------------------------------------------------------------------

if model is not None:
    rule("3. classify the ten shipped digits")
    print("A module can import and still generate wrong code for this core, and")
    print("nothing about that raises. Correct answers are the actual proof.")
    print()

    probs = array.array("f", (0.0 for _ in range(n_classes)))
    correct = 0
    total_us = 0
    worst_us = 0

    for digit in range(10):
        with open(DEV + "/mnist_%d.bin" % digit, "rb") as handle:
            img = array.array("B", handle.read())

        t0 = time.ticks_us()
        model.run(img, probs)
        dt = time.ticks_diff(time.ticks_us(), t0)

        best, best_p = 0, probs[0]
        for i in range(1, n_classes):
            if probs[i] > best_p:
                best, best_p = i, probs[i]

        ok = best == digit
        correct += ok
        total_us += dt
        if dt > worst_us:
            worst_us = dt
        print("  digit %d -> %d  p=%.3f  %6d us  %s"
              % (digit, best, best_p, dt, "ok" if ok else "WRONG"))
        del img

    print()
    print("  %d/10 correct" % correct)
    print("  mean %d us, worst %d us over 51228 MACs" % (total_us // 10, worst_us))
    print("  -> %.2f cycles per MAC at %d MHz"
          % ((total_us / 10.0) * 1e-6 * CLOCK_HZ / 51228, CLOCK_HZ // 1000000))
    if correct < 9:
        print()
        print("  IMPORT SUCCEEDED BUT THE ARITHMETIC IS WRONG. That is worse than")
        print("  a failed import, because it fails silently. Do not use this path.")


# ---------------------------------------------------------------------------
# 3b. The same two questions, for the keyword model
# ---------------------------------------------------------------------------
#
# Optional, and skipped silently when the files are absent, so this script stays
# useful before there is a trained model to run.
#
# Section 3 proves the *port* works, using a model whose answers are known
# upstream. This proves the port works *on our model*, which is a different
# claim: an architecture within TinyMaix's stated limits can still hit one of
# the paths nobody exercises. And it is the only way to get a real inference
# time for the thing that will actually ship.
#
# Expected classes ride in the filenames -- `kw_<class>_<n>.bin` -- for the same
# reason the MNIST digits do: it needs no schema shared between two agents, and
# a file that gets renamed is obviously wrong rather than quietly mismatched.
# The scores come from the host TFLite interpreter on the identical bytes, so a
# disagreement localises to the device rather than to the model.

if model is not None:
    try:
        names = [f for f in os.listdir(DEV) if f.startswith("kw_") and f.endswith(".bin")]
    except Exception:      # noqa: BLE001
        names = []

    if names:
        rule("3b. classify the keyword patches")
        kw_model = None
        try:
            with open(DEV + "/model.tmdl", "rb") as handle:
                blob = handle.read()
            before = heap()
            data = array.array("B", blob)
            del blob
            gc.collect()
            kw_model = cnn_mod.new(data)
            del data
            after = heap()
            dims = kw_model.output_dimensions()
            kw_classes = dims[0]
            print("  model.tmdl loaded, %d classes, %d B resident"
                  % (kw_classes, before - after))
            print()

            scores = array.array("f", (0.0 for _ in range(kw_classes)))
            names.sort()
            agree = 0
            total_us = 0
            for name in names:
                want = int(name.split("_")[1])
                with open(DEV + "/" + name, "rb") as handle:
                    patch = array.array("B", handle.read())
                t0 = time.ticks_us()
                kw_model.run(patch, scores)
                dt = time.ticks_diff(time.ticks_us(), t0)
                total_us += dt

                best, best_p = 0, scores[0]
                second = -1.0
                for i in range(1, kw_classes):
                    if scores[i] > best_p:
                        second = best_p
                        best, best_p = i, scores[i]
                    elif scores[i] > second:
                        second = scores[i]

                ok = best == want
                agree += ok
                print("    %-22s -> %2d (want %2d) p=%.4f margin %.4f  %6d us  %s"
                      % (name, best, want, best_p, best_p - second, dt,
                         "ok" if ok else "DISAGREES WITH HOST"))
                del patch

            print()
            print("  %d/%d agree with the host interpreter" % (agree, len(names)))
            print("  mean inference %d us (%.1f ms)"
                  % (total_us // len(names), total_us / 1000.0 / len(names)))
            print("  DTW on this board measures 616-672 ms of matching, so the")
            print("  break-even is ~600 ms and ~200 ms is a clear win.")
            if agree < len(names):
                print()
                print("  A disagreement here with section 3 passing means the port is")
                print("  fine and this model reaches a path the MNIST one does not.")
                print("  Send the failing patch and the host scores back to si-model.")
        except Exception as exc:      # noqa: BLE001
            print("  FAILED: %s: %s" % (type(exc).__name__, exc))
        finally:
            kw_model = None
            gc.collect()


# ---------------------------------------------------------------------------
# 4. The fallback: int8 convolution in viper
# ---------------------------------------------------------------------------
#
# Two things this has to get right, and they are the same two things the
# existing viper front end had to get right.
#
# **ptr8 loads are zero-extended**, exactly as ptr16 loads are (docs/speech.md,
# "What the viper port cost to get right"). Sign-extending each one costs
# statements in the innermost loop there is. So activations and weights are both
# stored *biased by +128* as uint8 and the inner loop is a plain unsigned dot
# product, with the bias unwound algebraically afterwards:
#
#     sum (Wu-128)(Xu-128) = sum Wu*Xu - 128*sum Xu - 128*sum Wu + 128*128*n
#
# and since `sum Wu = sum W + 128*n`, the last two terms collapse to exactly
# `-128 * sum W` over the *signed* weights. So the whole input-independent half
# of the correction is one number per filter, folded into the bias when the
# weights are packed -- there is no `128*128*n` term to carry, and adding one
# leaves every accumulator wrong by 1179648 without crashing. See
# `tools/test_conv_int8.py`, which is what found that.
#
# `sum Xu` is per input patch but shared across every output channel, so it is
# one extra pass amortised over `cho`. This is the same trick the DTW inner loop
# already uses on templates, where the +32768 bias cancels in a difference.
#
# **viper wraps at int32, silently**, and neither CPython nor MicroPython
# bytecode will reproduce it, because both carry unbounded ints. So the requant
# is bounded by construction rather than by hope: the accumulator is shifted
# right into 16 bits before it meets the per-channel multiplier, and that
# multiplier is Q14 rather than Q15. The product is then at most
# 65535 * 16383 = 1073659905, **50.00% of int32** -- the same factor of two the
# FFT twiddle keeps, and for the same reason. Q15 would put it at
# 65535 * 32767 = 2147385345, **100.00%**, which is not a margin.
#
# The bound on the accumulator itself: each term is (W-128)(X-128) with both
# factors in [-128, 127], so |term| <= 16384, and |acc| <= 16384 * kh*kw*chi.
# For the widest layer contemplated (kh*kw*chi = 2304, TinyMaix's own ceiling)
# that is 37.7e6 -- 1.76% of int32, 56x clear. The accumulator is not the risk;
# the requant multiply is, which is why that is the one that is bounded.

rule("4. fallback: hand-rolled int8 convolution in viper")

try:
    import micropython

    @micropython.viper
    def _dot_u8(x: ptr8, w: ptr8, n: int) -> int:
        acc = 0
        i = 0
        while i < n:
            acc += int(x[i]) * int(w[i])
            i += 1
        return acc

    @micropython.viper
    def _dot_u8x4(x: ptr8, w: ptr8, n: int) -> int:
        acc = 0
        i = 0
        while i < n:
            acc += int(x[i]) * int(w[i])
            acc += int(x[i + 1]) * int(w[i + 1])
            acc += int(x[i + 2]) * int(w[i + 2])
            acc += int(x[i + 3]) * int(w[i + 3])
            i += 4
        return acc

    @micropython.viper
    def _conv2d_u8(src: ptr8, dst: ptr8, wts: ptr8, cfg: ptr32) -> int:
        """Valid-padding strided conv2d, uint8 in and out, int32 accumulator.

        Four arguments because viper's native calling convention is happiest
        with few, so everything scalar rides in `cfg`, and the per-channel
        bias / multiplier / shift tables ride after it in the same int32 array.

        Returns the peak |acc >> rsh| seen, so the caller can check the bound
        that the comment above proves rather than trusting it.
        """
        # Every ptr32 load is wrapped in int(), which is what types it *signed*
        # -- docs/speech.md, "What the viper port cost to get right". The shapes
        # would survive being read unsigned; the per-channel biases below would
        # not, and they are read through the same pointer.
        in_w = int(cfg[1])
        chi = int(cfg[2])
        out_h = int(cfg[3])
        out_w = int(cfg[4])
        cho = int(cfg[5])
        kh = int(cfg[6])
        kw = int(cfg[7])
        sh = int(cfg[8])
        sw = int(cfg[9])
        rsh = int(cfg[10])
        relu = int(cfg[11])

        kwc = kw * chi          # one patch row: contiguous in HWC
        krow = in_w * chi       # source row stride
        ksz = kh * kwc          # weights per output channel
        b_off = 12
        m_off = b_off + cho
        s_off = m_off + cho

        peak = 0
        o = 0
        oy = 0
        while oy < out_h:
            ox = 0
            while ox < out_w:
                base = (oy * sh) * krow + (ox * sw) * chi

                # sum of the patch, shared by every output channel
                psum = 0
                r = 0
                sp = base
                while r < kh:
                    i = 0
                    while i < kwc:
                        psum += int(src[sp + i])
                        i += 1
                    sp += krow
                    r += 1
                psum = psum << 7        # the -128 * sum X term

                oc = 0
                wp = 0
                while oc < cho:
                    acc = 0
                    r = 0
                    sp = base
                    wq = wp
                    while r < kh:
                        i = 0
                        while i < kwc:
                            acc += int(src[sp + i]) * int(wts[wq + i])
                            i += 1
                        sp += krow
                        wq += kwc
                        r += 1
                    acc = acc - psum + int(cfg[b_off + oc])
                    if relu != 0:
                        if acc < 0:
                            acc = 0
                    a = acc >> rsh
                    p = a
                    if p < 0:
                        p = -p
                    if p > peak:
                        peak = p
                    v = (a * int(cfg[m_off + oc])) >> int(cfg[s_off + oc])
                    v += 128
                    if v < 0:
                        v = 0
                    if v > 255:
                        v = 255
                    dst[o] = v
                    o += 1
                    oc += 1
                    wp += ksz
                ox += 1
            oy += 1
        return peak

    # --- inner loop, in isolation ---------------------------------------
    N = 2048
    xb = bytearray(N)
    wb = bytearray(N)
    for i in range(N):
        xb[i] = (i * 37) & 0xFF
        wb[i] = (i * 91) & 0xFF

    for name, fn in (("plain", _dot_u8), ("4x unrolled", _dot_u8x4)):
        t0 = time.ticks_us()
        for _ in range(50):
            fn(xb, wb, N)
        dt = time.ticks_diff(time.ticks_us(), t0)
        per = dt / (50.0 * N)
        print("  uint8 dot, %-12s %8.4f us/MAC  %5.1f cycles/MAC"
              % (name, per, per * 1e-6 * CLOCK_HZ))
    del xb, wb
    gc.collect()

    # --- a whole layer, gather and requant included ----------------------
    #
    # Topology C from the host-side budget: a first convolution spanning the
    # full mel axis, so everything after it is one-dimensional in time. Input
    # is 64 frames x 24 bands x 1.
    print()
    print("  whole layers, 64x24x1 input (gather + MAC + requant):")

    LAYERS = (
        # (in_h, in_w, chi, kh, kw, cho, stride_h, stride_w)
        (64, 24, 1, 3, 24, 16, 1, 1),
        (62, 1, 16, 3, 1, 24, 2, 1),
        (30, 1, 24, 3, 1, 32, 2, 1),
    )

    grand_us = 0
    grand_macs = 0
    for (ih, iw, chi, kh, kw, cho, sh, sw) in LAYERS:
        oh = (ih - kh) // sh + 1
        ow = (iw - kw) // sw + 1
        n_terms = kh * kw * chi
        macs = oh * ow * cho * n_terms

        src = bytearray(ih * iw * chi)
        dst = bytearray(oh * ow * cho)
        wts = bytearray(cho * n_terms)

        # rsh chosen from the algebra so |acc >> rsh| < 65536, which is what
        # keeps the requant multiply at half of int32.
        bound = 16384 * n_terms
        rsh = 0
        while (bound >> rsh) >= 65536:
            rsh += 1

        cfg = array.array("i", [0] * (12 + 3 * cho))
        cfg[0] = ih
        cfg[1] = iw
        cfg[2] = chi
        cfg[3] = oh
        cfg[4] = ow
        cfg[5] = cho
        cfg[6] = kh
        cfg[7] = kw
        cfg[8] = sh
        cfg[9] = sw
        cfg[10] = rsh
        cfg[11] = 1
        for c in range(cho):
            cfg[12 + cho + c] = 16383       # Q14 multiplier, at its maximum
            cfg[12 + 2 * cho + c] = 14

        def fill(value, signed):
            """Set both buffers to one uint8 value and fold the matching bias.

            The bias carries `-128 * sum(W)` over the *signed* weights, which
            is the whole input-independent half of unwinding the +128 bias.
            Leaving it at zero does not crash; it leaves every accumulator
            wrong, and wrong in the direction that makes the bound look
            violated when it is not. tools/test_conv_int8.py has the algebra.
            """
            for i in range(len(src)):
                src[i] = value
            for i in range(len(wts)):
                wts[i] = value
            for ch in range(cho):
                cfg[12 + ch] = -128 * (signed * n_terms)

        # Pass 1, timing: operands at full width, which is what a real layer
        # looks like to the multiplier.
        fill(255, 127)
        t0 = time.ticks_us()
        _conv2d_u8(src, dst, wts, cfg)
        dt = time.ticks_diff(time.ticks_us(), t0)
        grand_us += dt
        grand_macs += macs

        # Pass 2, the bound: every product at +16384, which needs both factors
        # at -128 and is the true algebraic maximum. It is easy to reach for
        # +127 instead and land on 16129, 1.6% short, and never know.
        #
        # This has to run on the device and not only on the host, because
        # viper's int is a 32-bit machine word that wraps silently while
        # CPython's is unbounded -- so the host test proves the algebra and
        # this proves the port.
        fill(0, -128)
        peak = _conv2d_u8(src, dst, wts, cfg)

        occupancy = 100.0 * peak * 16383 / 2147483648.0
        print("    %2dx%-2d x%-3d k%dx%-2d -> %2dx%-2d x%-3d  %6d MAC  %6d us"
              % (ih, iw, chi, kh, kw, oh, ow, cho, macs, dt))
        print("       kh*kw*chi=%-5d rshift=%-2d peak |acc>>r| = %d of 65536,  "
              "requant product %.2f%% of int32 %s"
              % (n_terms, rsh, peak, occupancy,
                 "" if peak < 65536 else "** BOUND VIOLATED **"))
        if peak != (bound >> rsh):
            print("       expected peak %d from the algebra -- a difference here "
                  "means viper wrapped where the host did not"
                  % (bound >> rsh))
        del src, dst, wts, cfg
        gc.collect()

    print()
    print("  whole network: %d MAC in %d us (%.1f ms), %.1f cycles/MAC end to end"
          % (grand_macs, grand_us, grand_us / 1000.0,
             grand_us * 1e-6 * CLOCK_HZ / grand_macs))
    print("  for comparison, DTW against the 66-template set measures 616-672 ms")

except Exception as exc:      # noqa: BLE001
    print("viper section FAILED: %s: %s" % (type(exc).__name__, exc))


# ---------------------------------------------------------------------------
# 5. Does it still fit alongside everything else
# ---------------------------------------------------------------------------

rule("5. coexistence")

print("Sections 1-3 ran on an empty heap, which is not the heap the program has.")
print()

gc.collect()
print("free now                        %7d B" % gc.mem_free())

held = []
try:
    # Capture buffer first, and as listen.allocate_samples actually builds it:
    # array('h', bytearray(2*n)) holds both objects at once and so spikes to
    # 188 KB to end up with 94. docs/speech.md is emphatic that ordering by
    # transient peak rather than by resident size is what makes that spike free.
    held.append(array.array("h", bytearray(2 * 48000)))
    gc.collect()
    print("+ 94 KB capture buffer          %7d B free" % gc.mem_free())

    held.append(bytearray(37 * 1024))
    gc.collect()
    print("+ 37 KB ELIZA rules             %7d B free" % gc.mem_free())

    held.append(bytearray(5000))
    gc.collect()
    print("+ 5 KB framebuffer              %7d B free" % gc.mem_free())
    print("  largest contiguous           %7d B" % biggest())

    if model is not None:
        probs = array.array("f", (0.0 for _ in range(n_classes)))
        with open(DEV + "/mnist_0.bin", "rb") as handle:
            img = array.array("B", handle.read())
        t0 = time.ticks_us()
        model.run(img, probs)
        dt = time.ticks_diff(time.ticks_us(), t0)
        print("  inference under pressure     %7d us" % dt)
        del img, probs

    # And the transitional case, where the DTW template set is still resident
    # because it stays as the fallback.
    held.append(bytearray(140064))
    gc.collect()
    print("+ 137 KB DTW templates          %7d B free" % gc.mem_free())
    print("  largest contiguous           %7d B" % biggest())

except MemoryError as exc:
    print("  MemoryError at this step: %s" % exc)
finally:
    del held
    gc.collect()
    print()
    print("free after release              %7d B" % gc.mem_free())


# ---------------------------------------------------------------------------
# 6. Confirm or refute the scratch-buffer overrun, on real silicon
# ---------------------------------------------------------------------------
#
# Opt-in: runs only when `/cnn/overrun.tmdl` is present, because it deliberately
# provokes a heap overrun and the board should be reset afterwards.
#
# The claim under test, from reading emlearn's `mod_cnn.c` and TinyMaix's
# `tm_model.c`: `data_buffer` is allocated with the *file length* and handed to
# `tm_load()` as the activation buffer, and `tm_load()` never compares it against
# the model's own `buf_size`. So a model with `buf_size > file_size` writes past
# the end of a heap allocation with nothing raised.
#
# That is proved from source but not from hardware, and it is worth the
# difference: the whole point of the finding is that it produces no symptom at
# the point of failure, so "we ran it and nothing happened" is exactly what the
# claim predicts. A canary is the only thing that turns it into an observation.
#
# Method. After the model is loaded, fill **every remaining free byte** with
# bytearrays of a known pattern, then run inference, then check the pattern.
# With no free heap left, any write outside `data_buffer` that lands in
# unallocated memory must land in a canary. Filling first and freeing a hole
# would be the obvious approach and is worse -- it leaves the overrun somewhere
# it can land harmlessly, and a null result would then mean nothing.
#
# A null result here is still weaker than a positive one, and is reported as
# such: the overrun could have landed inside another live object rather than in
# free space.

PATTERN = 0xA5

try:
    have_overrun = "overrun.tmdl" in os.listdir(DEV)
except Exception:      # noqa: BLE001
    have_overrun = False

if model is not None and have_overrun:
    rule("6. scratch-buffer overrun: canary test")
    print("This deliberately provokes a heap overrun. RESET THE BOARD AFTER IT.")
    print()

    victim = None
    canaries = []
    try:
        with open(DEV + "/overrun.tmdl", "rb") as handle:
            blob = handle.read()
        # The header's buf_size sits at offset 12, sub_size at 16. Read them
        # here rather than trusting the host: the number that matters is the one
        # this board will act on.
        buf_size = int.from_bytes(blob[12:16], "little")
        sub_size = int.from_bytes(blob[16:20], "little")
        need = buf_size + sub_size
        print("  file        %d B" % len(blob))
        print("  buf_size    %d B  (+ %d sub)" % (buf_size, sub_size))
        print("  shortfall   %d B" % (need - len(blob)))
        if need <= len(blob):
            print("  -- this model does NOT overrun; nothing to demonstrate.")
        else:
            data = array.array("B", blob)
            del blob
            gc.collect()
            victim = cnn_mod.new(data)
            del data
            gc.collect()
            dims = victim.output_dimensions()
            scores = array.array("f", (0.0 for _ in range(dims[0])))
            for name in os.listdir(DEV):
                if name.startswith("kw_") and name.endswith(".bin"):
                    with open(DEV + "/" + name, "rb") as handle:
                        patch = array.array("B", handle.read())
                    break
            else:
                patch = None

            if patch is None:
                print("  no kw_*.bin patch to feed it; skipping")
            else:
                # Fill every free byte with the pattern.
                budget = gc.mem_free()
                filled = 0
                block = 2048
                while filled < budget:
                    try:
                        buf = bytearray(block)
                        for i in range(len(buf)):
                            buf[i] = PATTERN
                        canaries.append(buf)
                        filled += block
                    except MemoryError:
                        if block <= 32:
                            break
                        block >>= 1
                print("  canaries    %d blocks, %d B, %d B free after"
                      % (len(canaries), filled, gc.mem_free()))

                victim.run(patch, scores)

                dirty_blocks = 0
                dirty_bytes = 0
                for buf in canaries:
                    hit = 0
                    for i in range(len(buf)):
                        if buf[i] != PATTERN:
                            hit += 1
                    if hit:
                        dirty_blocks += 1
                        dirty_bytes += hit
                print()
                if dirty_bytes:
                    print("  CONFIRMED: %d bytes across %d canary blocks were "
                          "overwritten" % (dirty_bytes, dirty_blocks))
                    print("  by an inference that raised nothing and returned "
                          "normally.")
                    print("  Expected shortfall was %d B." % (need - len(blob)))
                else:
                    print("  No canary was touched. That is NOT a refutation:")
                    print("  the overrun may have landed inside another live")
                    print("  object rather than in the free space this filled.")
                    print("  Inconclusive -- the static check stands either way.")
    except Exception as exc:      # noqa: BLE001
        print("  FAILED: %s: %s" % (type(exc).__name__, exc))
    finally:
        del canaries
        victim = None
        gc.collect()
        print()
        print("  RESET THE BOARD NOW: uvx mpremote connect <port> reset")


rule("summary")
if import_error is not None:
    print("emlearn_cnn_int8 does NOT import on this board: %s" % import_error)
    print("-> the trained-model path is the viper convolution in section 4.")
elif model is None:
    print("emlearn_cnn_int8 imports but would not load a model.")
    print("-> report the exact error; it decides whether this is fixable.")
else:
    print("emlearn_cnn_int8 imports, loads and classifies on this board.")
    print("-> the model may use anything TinyMaix supports, within the bounds")
    print("   tools/tmdl_info.py checks.")
