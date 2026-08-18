#!/usr/bin/env python3
"""Inspect a TinyMaix `.tmdl` model, and refuse the two ways it corrupts the heap.

    python3 tools/tmdl_info.py model.tmdl [--pad]

`emlearn_cnn_int8` is the only trained-model path that stays inside MicroPython
on this board, and it is a thin wrapper around Sipeed's TinyMaix. Thin enough
that two of TinyMaix's assumptions are the *caller's* problem, and neither one
produces an error when it is violated:

**1. The activation scratch buffer is sized by guesswork.** `mod_cnn_new()`
allocates `data_buffer` with the length of the model *file* and hands it to
`tm_load()` as the ping-pong buffer for every intermediate activation.
`tm_load()` does no size check against the model's own `buf_size` field -- it
only allocates for itself when passed NULL, and it is never passed NULL here.
So a model whose activations are larger than its weights writes past the end of
a heap allocation, on a heap that also holds the ELIZA rules, the 94 KB capture
buffer and the framebuffer. There is no exception and no error return; the
symptom is arbitrary later corruption.

That combination -- small weights, large activations -- is not exotic. It is
what you get from exactly the topology anyone would reach for first: a few
strided convolutions with modest channel counts over a large input. The MNIST
model emlearn ships is the other way round (3912 bytes of file against a
2136-byte buffer) which is precisely why nobody upstream has hit this.

The fix is `--pad`: append zero bytes until the file is at least `buf_size`
long. Nothing reads them -- `tm_load()` reaches the layers through the offsets
in the header -- but `data_buffer` is sized from the file length, so padding the
file is what buys the scratch space. Cheap, and it is the only lever the caller
has without patching the module.

**2. Two of TinyMaix's compile-time array bounds are unchecked.**
`tml_conv2d_dwconv2d()` tests `kw*kh > TM_MAX_KSIZE` and returns `TM_ERR_KSIZE`,
which is a clean failure. It does *not* test `sbuf`, whose 2304 entries hold
`kh*kw*in_channels` per output pixel, nor `sumscale`, whose 1000 entries are
filled one per output channel. Both are static arrays inside the native module.
Both overrun silently.

So this checks all three bounds against the values baked into the prebuilt
module, and exits non-zero if any of them is violated. Run it on every `.tmdl`
before it goes near the board.

Reference: the format is `tm_mdlbin_t` and `tml_head_t` in TinyMaix's
`include/tinymaix.h`; the limits are in emlearn-micropython's
`src/tinymaix_cnn/int8/tm_port.h`; the buffer sizing is `cal_buf_size()` in
TinyMaix's `tools/tflite2tmdl.py`.

Verified against emlearn's own `examples/mnist_cnn/mnist_cnn_int8.tmdl`, which
parses to six layers and a 2136-byte buffer and walks exactly to end of file.
"""

import os
import struct
import sys

# Baked into the prebuilt module at compile time; see tm_port.h. Changing these
# means building your own .mpy, which is not a thing this project does.
TM_MAX_KSIZE = 5 * 5          # kw*kh -- checked by TinyMaix, clean failure
TM_MAX_KCSIZE = 3 * 3 * 256   # kw*kh*chi -- NOT checked, overruns sbuf
TM_MAX_CSIZE = 1000           # output channels -- NOT checked, overruns sumscale

MDL_TYPES = {0: "int8", 1: "int16", 2: "fp32", 3: "fp16"}
LAYER_TYPES = {
    0: "CONV2D",
    1: "GAP",
    2: "FC",
    3: "SOFTMAX",
    4: "RESHAPE",
    5: "DWCONV2D",
    6: "ADD",
}

HEADER_BYTES = 64
LAYER_HEAD_BYTES = 48


class Layer:
    """One `tml_head_t`, plus the conv fields when the type carries them."""

    def __init__(self, index, kind, is_out, size, in_oft, out_oft,
                 in_dims, out_dims, in_s, in_zp, out_s, out_zp, conv=None):
        self.index = index
        self.kind = kind
        self.is_out = is_out
        self.size = size
        self.in_oft = in_oft
        self.out_oft = out_oft
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.in_s = in_s
        self.in_zp = in_zp
        self.out_s = out_s
        self.out_zp = out_zp
        self.conv = conv  # dict for CONV2D/DWCONV2D, else None

    @property
    def in_elems(self):
        return self.in_dims[1] * self.in_dims[2] * self.in_dims[3]

    @property
    def out_elems(self):
        return self.out_dims[1] * self.out_dims[2] * self.out_dims[3]

    @property
    def macs(self):
        if self.conv is None:
            if self.kind == "FC":
                return self.in_elems * self.out_elems
            return 0
        k = self.conv["kernel_h"] * self.conv["kernel_w"]
        chi = 1 if self.conv["depth_mul"] else self.in_dims[3]
        return self.out_elems * k * chi


def parse(data):
    """Return (header dict, [Layer]). Raises ValueError on anything unexpected."""
    if len(data) < HEADER_BYTES:
        raise ValueError("file is shorter than the %d-byte header" % HEADER_BYTES)

    magic, mdl_type, out_deq, in_cnt, out_cnt, layer_cnt, buf_size, sub_size = \
        struct.unpack_from("<4sBBHHHII", data, 0)
    if magic != b"MAIX":
        raise ValueError("bad magic %r -- not a .tmdl (expected b'MAIX')" % (magic,))

    head = {
        "mdl_type": mdl_type,
        "out_deq": out_deq,
        "input_cnt": in_cnt,
        "output_cnt": out_cnt,
        "layer_cnt": layer_cnt,
        "buf_size": buf_size,
        "sub_size": sub_size,
        "in_dims": struct.unpack_from("<4H", data, 20),
        "out_dims": struct.unpack_from("<4H", data, 28),
    }

    layers = []
    off = HEADER_BYTES
    for i in range(layer_cnt):
        if off + LAYER_HEAD_BYTES > len(data):
            raise ValueError("layer %d head runs past end of file" % i)
        kind_id, is_out, size, in_oft, out_oft = struct.unpack_from("<HHIII", data, off)
        in_dims = struct.unpack_from("<4H", data, off + 16)
        out_dims = struct.unpack_from("<4H", data, off + 24)
        in_s, in_zp, out_s, out_zp = struct.unpack_from("<fifi", data, off + 32)

        kind = LAYER_TYPES.get(kind_id, "UNKNOWN(%d)" % kind_id)
        conv = None
        scales = None
        if kind == "FC":
            # pack_fc() lays the body out as ws_oft, w_oft, b_oft, reserve --
            # four uint32 -- then the per-output-channel weight scales. The
            # offsets are from the start of the layer. Read the scales, because
            # whether they are all equal decides whether this model works; see
            # check() for why.
            ws_oft, = struct.unpack_from("<I", data, off + LAYER_HEAD_BYTES)
            n_out = out_dims[3]
            scales = struct.unpack_from("<%df" % n_out, data, off + ws_oft)
        if kind in ("CONV2D", "DWCONV2D"):
            kw, kh, sw, sh, dw, dh, act = struct.unpack_from("<6BH", data, off + 48)
            pad = struct.unpack_from("<4B", data, off + 56)
            depth_mul, = struct.unpack_from("<I", data, off + 60)
            conv = {
                "kernel_w": kw, "kernel_h": kh,
                "stride_w": sw, "stride_h": sh,
                "dilation_w": dw, "dilation_h": dh,
                "act": act, "pad": pad, "depth_mul": depth_mul,
            }

        lay = Layer(i, kind, is_out, size, in_oft, out_oft,
                    in_dims, out_dims, in_s, in_zp, out_s, out_zp, conv)
        lay.scales = scales
        layers.append(lay)
        if size <= 0:
            raise ValueError("layer %d has size %d -- cannot walk further" % (i, size))
        off += size

    if off > len(data):
        raise ValueError(
            "layer walk ran %d bytes past the end of a %d-byte file -- the "
            "layer sizes do not describe this file" % (off - len(data), len(data)))

    # Trailing bytes are legitimate: `--pad` appends them precisely so that
    # emlearn's `data_buffer = len(file)` is large enough for the activations.
    # They are never read -- tm_load reaches the layers through the offsets in
    # the header. But they are reported, because trailing bytes that nobody
    # added on purpose would mean the parser and the file disagree.
    head["padding"] = len(data) - off

    return head, layers


def fc_scale_kind(scales):
    """(number of real scales, is it per-tensor).

    `pack_fc()` reserves room for one float per output channel but writes only
    as many as the converter produced, zero-padding the rest. So a per-tensor
    dense layer appears as one scale followed by zeros, and a per-channel one as
    a full set. A quantisation scale is never zero, which is what separates
    padding from data -- there is no length field to read.

    The distinction matters because `tml_fc()` uses `ws[0]` and nothing else.
    Per-tensor is therefore correct and per-channel is silently wrong, which is
    the opposite of the usual advice about quantising dense layers.
    """
    n_real = 0
    for value in scales:
        if value == 0.0:
            break
        n_real += 1
    return n_real, n_real <= 1


def check(path, data, head, layers):
    """Return a list of problem strings. Empty means the model is safe to load."""
    problems = []

    need = head["buf_size"] + head["sub_size"]
    if need > len(data):
        problems.append(
            "activation buffer needs %d bytes but emlearn will only allocate %d "
            "(the file length) -- tm_load does not check, so this overruns the "
            "heap silently. Re-run with --pad." % (need, len(data)))

    if head["mdl_type"] == 0 and not head["out_deq"]:
        problems.append(
            "out_deq is 0 on an int8 model. TinyMaix then leaves the output as "
            "int8, but emlearn's run() reads it as float32 unconditionally "
            "(`output_buffer[i] = out.dataf[i]`), so the class scores come back "
            "as reinterpreted bytes -- garbage, with nothing raised. Re-run "
            "tflite2tmdl.py with out_deq = 1.")

    for lay in layers:
        if lay.kind != "FC" or not lay.scales:
            continue
        n_real, per_tensor = fc_scale_kind(lay.scales)
        if not per_tensor:
            problems.append(
                "layer %d (FC) carries %d per-channel weight scales, but "
                "tml_fc() reads ws[0] only and applies it to every output "
                "(`sum*in_s*ws[0]/out_s`). The converter writes them all and the "
                "runtime ignores all but the first, so every logit but one comes "
                "out scaled wrong -- silently, with no error. Quantise the dense "
                "layer PER-TENSOR. Scales run %.6g to %.6g." % (
                    lay.index, n_real, min(lay.scales), max(lay.scales)))

    if head["mdl_type"] != 0:
        problems.append(
            "mdl_type is %s; emlearn_cnn_int8 raises tm_load error on anything "
            "but int8" % MDL_TYPES.get(head["mdl_type"], head["mdl_type"]))

    outs = [lay for lay in layers if lay.is_out]
    if len(outs) != 1:
        problems.append(
            "%d output layers; the emlearn wrapper raises 'only 1 output "
            "supported'" % len(outs))
    elif outs[0].out_dims[0] != 1:
        problems.append(
            "output is %d-dimensional; the wrapper raises 'output must be 1d'"
            % outs[0].out_dims[0])

    for lay in layers:
        if lay.kind.startswith("UNKNOWN"):
            problems.append("layer %d is %s" % (lay.index, lay.kind))
        if lay.conv is None:
            continue
        c = lay.conv
        k = c["kernel_h"] * c["kernel_w"]
        chi = 1 if c["depth_mul"] else lay.in_dims[3]
        # dwconv gathers cho*maxk into sbuf, conv gathers chi*maxk.
        gather = (lay.out_dims[3] if c["depth_mul"] else chi) * k
        if k > TM_MAX_KSIZE:
            problems.append(
                "layer %d kernel %dx%d = %d > TM_MAX_KSIZE %d (TinyMaix returns "
                "TM_ERR_KSIZE -- clean failure, but it will not run)"
                % (lay.index, c["kernel_h"], c["kernel_w"], k, TM_MAX_KSIZE))
        if gather > TM_MAX_KCSIZE:
            problems.append(
                "layer %d gathers %d values into sbuf[%d] -- UNCHECKED overrun "
                "of a static array inside the native module"
                % (lay.index, gather, TM_MAX_KCSIZE))
        if lay.out_dims[3] > TM_MAX_CSIZE:
            problems.append(
                "layer %d has %d output channels, sumscale holds %d -- UNCHECKED "
                "overrun" % (lay.index, lay.out_dims[3], TM_MAX_CSIZE))
        if c["dilation_w"] != 1 or c["dilation_h"] != 1:
            problems.append(
                "layer %d is dilated (%d,%d); TinyMaix returns TM_ERR_TODO"
                % (lay.index, c["dilation_w"], c["dilation_h"]))

    return problems


def report(path, data, head, layers):
    print("%s -- %d bytes" % (path, len(data)))
    print("  model type    %s%s" % (MDL_TYPES.get(head["mdl_type"], head["mdl_type"]),
                                    ", output dequantised" if head["out_deq"] else ""))
    print("  input         %s" % (tuple(head["in_dims"][1:]),))
    print("  layers        %d" % head["layer_cnt"])
    print()

    macs = 0
    for lay in layers:
        extra = ""
        if lay.conv:
            c = lay.conv
            extra = "  k=%dx%d s=%dx%d%s" % (
                c["kernel_h"], c["kernel_w"], c["stride_h"], c["stride_w"],
                " dmul=%d" % c["depth_mul"] if c["depth_mul"] else "")
        macs += lay.macs
        print("  L%-2d %-8s %-16s -> %-16s %8d MAC%s"
              % (lay.index, lay.kind,
                 str(tuple(lay.in_dims[1:])), str(tuple(lay.out_dims[1:])),
                 lay.macs, extra))
    print()

    # The header's own out_dims field is not filled in by tflite2tmdl -- emlearn
    # ignores it too and walks the layers for the is_out flag instead. Do the same.
    outs = [lay for lay in layers if lay.is_out]
    if outs:
        print("  classes       %d" % outs[-1].out_dims[3])
    for lay in layers:
        if lay.kind == "FC" and lay.scales:
            n_real, per_tensor = fc_scale_kind(lay.scales)
            print("  L%d dense      %s weight quantisation (%d scale%s)"
                  % (lay.index, "per-tensor" if per_tensor else "PER-CHANNEL",
                     n_real, "" if n_real == 1 else "s"))
    print("  total MACs    %d" % macs)
    print("  activations   %d B  (ping-pong %d + ADD-buf %d)"
          % (head["buf_size"] + head["sub_size"], head["buf_size"], head["sub_size"]))
    print("  file          %d B%s" % (
        len(data),
        "  (%d of it trailing padding, so the activation scratch fits)"
        % head["padding"] if head["padding"] else ""))
    print("  heap at rest  %d B  (emlearn keeps model_buffer + data_buffer, both "
          "= file length)" % (2 * len(data)))
    print("  heap peak     %d B  (during new(), while the caller's array('B') is "
          "still alive)" % (3 * len(data)))
    print()

    # The input quantisation, and what it obliges the *host* to send.
    #
    # `tm_preprocess` is hardwired to TMPP_UINT2INT: the quantised value the
    # network sees is exactly `uint8 - 128`, and the model's own scale and zero
    # point are never consulted on that path. So this cannot be checked from
    # the file alone -- there is no wrong value of in_s here, only a host that
    # disagrees with it. What it can do is state the obligation in the units
    # the feature code works in, so the two can be compared by eye.
    #
    # Two recipes both satisfy it, and they are not interchangeable:
    #   in_s = 1/255, in_zp = -128  -- train on u8/255, i.e. reals in [0,1]
    #   in_s = 1,     in_zp = 0     -- train on reals in [-128,127] directly
    if layers:
        first = layers[0]
        lo = first.in_s * (-128 - first.in_zp)
        hi = first.in_s * (127 - first.in_zp)
        print("  input quant   scale %.8g, zero point %d" % (first.in_s, first.in_zp))
        print("                the network's real input range is [%.6g, %.6g]"
              % (lo, hi))
        print("                the wrapper sends quantised = uint8 - 128, so the")
        print("                host must emit uint8 = round(real / %.8g) + %d"
              % (first.in_s, 128 + first.in_zp))
        if abs(first.in_s - 1.0) < 1e-9 and first.in_zp == 0:
            print("                -> uint8 = int8_feature + 128, nothing else")
        elif abs(first.in_s - 1.0 / 255.0) < 1e-9 and first.in_zp == -128:
            print("                -> uint8 = round(255 * real), real in [0,1]")
        else:
            print("                -> neither of the two standard recipes; make")
            print("                   sure the feature code really does this")
    print()


def pad(path, data, head):
    need = head["buf_size"] + head["sub_size"]
    if need <= len(data):
        print("no padding needed: %d bytes of file for a %d-byte buffer"
              % (len(data), need))
        return 0
    extra = need - len(data)
    with open(path, "ab") as handle:
        handle.write(b"\x00" * extra)
    print("appended %d zero bytes: %d -> %d, now covers the %d-byte buffer"
          % (extra, len(data), len(data) + extra, need))
    return extra


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("-")]
    do_pad = "--pad" in argv[1:]
    if len(args) != 1:
        print(__doc__.strip().splitlines()[2].strip())
        return 2

    path = args[0]
    with open(path, "rb") as handle:
        data = handle.read()

    try:
        head, layers = parse(data)
    except ValueError as exc:
        print("%s: %s" % (path, exc))
        return 1

    report(path, data, head, layers)

    if do_pad:
        if pad(path, data, head):
            with open(path, "rb") as handle:
                data = handle.read()
            head, layers = parse(data)

    problems = check(path, data, head, layers)
    if problems:
        print("PROBLEMS -- do not load this on the board:")
        for p in problems:
            print("  - %s" % p)
        return 1

    print("OK: within every TinyMaix bound, and the activation buffer fits.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
