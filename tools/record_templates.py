#!/usr/bin/env python3
"""Turn recorded utterances into the frozen template set the device matches on.

    # record through the board, word by word, then build the templates
    uvx --from pyserial python tools/record_templates.py --board

    # or build from WAVs already captured (tools/enrol.py's output, or any dir)
    python3 tools/record_templates.py --from recordings/

**Enrolment goes through the board's own microphone. There is no Mac-microphone
path, deliberately.** An earlier version had one and it was a footgun: templates
built from a MacBook mic look perfectly healthy -- they endpoint cleanly, the
frame counts are sane, nothing warns -- and then match badly against recognition
through the ES8311, for a reason nothing in the output points at.

The mismatch is not something this recogniser can absorb. Speaker-dependent DTW
with three templates per word has no model to generalise with; a systematic
offset simply adds to every frame distance and squashes the margin between the
right word and the runner-up. Cepstral mean normalisation is the right tool for
the *convolutional* part of a channel, but it is estimated here from a single
~0.5 s word, and over one word the cepstral mean substantially **is** the
phonetic content -- subtracting it removes signal along with channel. And the
rest of the mismatch is not convolutional at all: the two microphones have
different noise floors (additive), and gain, AGC, clipping and the enclosure are
not linear time-invariant in the first place.

`tools/mic_margin.py` measures exactly how much that costs, on real audio,
rather than leaving it as an argument. Run it before trusting any template set.

Nothing is written to the board's filesystem: audio is captured to a RAM buffer
and streamed out over USB CDC by `src/record_stream.py`, read here through
`tools/pull_recording.py`. Flash is written only by an attended `./tools/deploy.sh`,
which is what that script does anyway.

Endpointing, MFCC and the packed layout all come from the same modules the
device uses, so what this writes is what the board compares against.

Output (see --format):

    py    src/templates.py    one bytes literal plus an index. What deploy.sh
                              already knows how to copy.
    bin   src/templates.bin   the same bytes, plus a small src/templates.py
          + src/templates.py  that reads it. Measurably better on both counts
                              below; use it if deploy.sh will carry two files.

**`--out` defaults to `src/` and `./tools/deploy.sh eliza` reads both files from
there.** Point it somewhere else for inspection by all means, but a template set
built into another directory will not deploy: the loader has to sit in `src/`
with the other device modules, and deploy.sh looks for the blob beside it
(`TEMPLATE_BLOB`). The two ends disagreed once -- the loader shipped, the blob
did not, and the board answered every press with a deflection, which is exactly
what a shy recogniser looks like. `tools/test_templates.py` now pins it.
"""

import argparse
import array
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

import mfcc    # noqa: E402
import vad     # noqa: E402
import vocab   # noqa: E402

# Named rather than written inline into add_argument so tools/test_templates.py
# can assert it against deploy.sh's TEMPLATE_BLOB instead of parsing this file.
DEFAULT_OUT = os.path.join(HERE, "..", "src")

REPS = 3
RECORD_SECONDS = 2.0
LEVEL_LOW = 2000      # peak below this and the word will sit in the noise


# --- Capture ---------------------------------------------------------------

def _pull():
    """tools/pull_recording.py, imported late so --from needs no pyserial."""
    import pull_recording
    return pull_recording


def level_note(st):
    """Advisory string for a capture, or None if it looks usable."""
    if st["peak"] >= 32767:
        return "clipping -- move back or drop the codec gain"
    if st["peak"] < LEVEL_LOW:
        return "too quiet -- move closer or raise the codec gain"
    if abs(st["mean"]) > 500:
        return "large DC offset (%d) -- codec may not have settled" % st["mean"]
    return None


def prompt_take(port, form, rep, seconds, rate):
    """Capture one utterance through the board. Returns frames, or None."""
    pull = _pull()
    while True:
        answer = input("  %-9s take %d of %d -- Enter to record, (s)kip: "
                       % (form.upper(), rep + 1, REPS)).strip().lower()
        if answer.startswith("s"):
            return None
        print("    recording %.1f s -- speak now" % seconds)
        try:
            got_rate, pcm, elapsed = pull.capture(port, seconds=seconds,
                                                  rate=rate)
        except pull.CaptureError as exc:
            print("    capture failed: %s" % exc)
            continue
        if got_rate != rate:
            print("    device recorded at %d Hz, not %d -- refusing"
                  % (got_rate, rate))
            continue

        samples = array.array("h")
        samples.frombytes(pcm)
        if sys.byteorder != "little":
            samples.byteswap()

        st = pull.stats(pcm)
        note = level_note(st)
        span = vad.endpoints(samples)
        if span is None:
            print("    peak %d, rms %.0f -- no utterance found. Again."
                  % (st["peak"], st["rms"]))
            continue
        ms = (span[1] - span[0]) / (rate / 1000.0)
        print("    peak %d, rms %.0f, %.0f ms of speech in %.2f s%s"
              % (st["peak"], st["rms"], ms, elapsed,
                 "  [%s]" % note if note else ""))
        if note:
            print("    Again.")
            continue
        return mfcc.mfcc_q8(samples[span[0]:span[1]])


def takes_from_board(forms, seconds, port_name, rate):
    pull = _pull()
    print("Recording %d forms x %d takes through the board's ES8311.\n"
          "Speak at the distance you will actually use it from.\n"
          % (len(forms), REPS))
    out = {}
    with pull.open_port(port_name, timeout=0.5) as port:
        # Round robin rather than three takes of one word back to back: three
        # consecutive takes are three copies of one delivery, and a template
        # set wants the variation.
        for rep in range(REPS):
            for form in forms:
                frames = prompt_take(port, form, rep, seconds, rate)
                if frames is not None:
                    out.setdefault(form, []).append(frames)
    return out


def form_from_filename(fn):
    """`mother_01.wav`, `mother.0.wav` and `mother.wav` all mean MOTHER.

    tools/enrol.py writes the first and tools/say_corpus.py the second, and a
    directory assembled by hand tends to be the third.
    """
    stem = fn.rsplit(".", 1)[0]
    for sep in ("_", "."):
        head, found, tail = stem.rpartition(sep)
        if found and tail.isdigit():
            stem = head
            break
    return stem.lower()


def takes_from_dir(path):
    """Read tools/enrol.py's manifest, or failing that `<label>_<rep>.wav`."""
    manifest = os.path.join(path, "manifest.json")
    files = []
    if os.path.exists(manifest):
        with open(manifest) as fh:
            doc = json.load(fh)
        for entry in doc.get("entries", []):
            files.append((entry["label"].lower(),
                          os.path.join(path, entry["file"])))
        print("manifest: %d entries, source %r"
              % (len(files), doc.get("source", "unknown")))
    else:
        for fn in sorted(os.listdir(path)):
            if not fn.endswith(".wav"):
                continue
            files.append((form_from_filename(fn), os.path.join(path, fn)))
        print("no manifest; took %d WAVs by filename" % len(files))

    out = {}
    dropped = []
    for form, wav in files:
        if form not in vocab.FORMS:
            dropped.append((form, "not in the vocabulary"))
            continue
        samples = vad.read_wav(wav)
        trimmed = vad.trim(samples)
        if trimmed is None:
            dropped.append((form, "endpointing found no utterance in %s"
                            % os.path.basename(wav)))
            continue
        out.setdefault(form, []).append(mfcc.mfcc_q8(trimmed))
    for form, why in dropped:
        print("  dropped %s: %s" % (form, why))
    return out


# --- Emission --------------------------------------------------------------

def _escape(data, width=24):
    lines = []
    for i in range(0, len(data), width):
        lines.append("    b'" + "".join("\\x%02x" % b for b in data[i:i + width])
                     + "'")
    return "\n".join(lines)


HEADER = '''"""Enrolled DTW templates. Generated -- do not edit.

Written by tools/record_templates.py. Speaker-dependent and
channel-dependent: these are one person at one microphone, and they do not
transfer to another of either.

Front end: %s
Recorded:  %s
Source:    %s
Packing:   %s

INDEX is (label, frame_offset, n_frames). Frame offsets, not byte offsets,
because the two differ before and after expansion.

A class may hold more than one spoken form (SAD holds SAD and SICK), so a
label appears once per template and the matcher takes the minimum over all of
them. See src/vocab.py.
"""

FORMAT = %d
N_CEPS = %d
N_FEAT = %d
FRAME_BYTES = %d
PACKED = %r
TOTAL_FRAMES = %d

# Allocate this much, once, and early. It is the largest single allocation the
# program makes, and MicroPython's heap does not compact -- see
# sounds.allocate_bytes for the measurement that taught this project the rule.
BUFFER_BYTES = %d

# What is actually on the filesystem, which is what readinto() fills.
BLOB_BYTES = %d

# Scratch needed by the expansion pass: the longest template's statics.
SCRATCH_BYTES = %d
'''

LOADER_DOC = '''    """Fill `buf` with match-ready templates, allocating it if not given.

    **This module does not keep a reference.** The returned buffer is the only
    one; whatever holds it must go on holding it for the life of the program,
    or the templates are collected at some unrelated later moment and the
    spotter reads freed memory.

    `bytearray(n)` is one allocation. Do not build this as
    `array("h", bytearray(n))` -- that holds both at once and peaks at twice
    the size, which is exactly how a 140 KB clip failed to allocate with
    174 KB free (see sounds.allocate_bytes).

    With PACKED == "statics" the blob is 12-wide Q8 statics and `expand` is
    **required**: it is called as `expand(buf, INDEX)` and must rewrite the
    buffer into 24-wide features. Passing nothing raises, deliberately.
    Matching against unexpanded statics does not crash -- it reads half a
    template and a trailing block of zeros, and returns confident nonsense.
    See docs/speech.md.
    """'''

LOADER_BIN = '''BLOB_FILE = "templates.bin"


def load(buf=None, expand=None):
%s
    if PACKED == "statics" and expand is None:
        raise ValueError("templates are packed as statics; load(buf, expand=)"
                         " needs the expansion pass")
    if buf is None:
        buf = bytearray(BUFFER_BYTES)
    with open(BLOB_FILE, "rb") as fh:
        got = fh.readinto(memoryview(buf)[0:BLOB_BYTES])
    if got != BLOB_BYTES:
        raise ValueError("templates.bin is %%d bytes, expected %%d"
                         %% (got, BLOB_BYTES))
    if PACKED == "statics":
        expand(buf, INDEX)
    return buf
''' % LOADER_DOC

LOADER_PY = '''

def load(buf=None, expand=None):
%s
    if PACKED == "statics" and expand is None:
        raise ValueError("templates are packed as statics; load(buf, expand=)"
                         " needs the expansion pass")
    if buf is None:
        buf = bytearray(BUFFER_BYTES)
    buf[0:len(BLOB)] = BLOB
    if PACKED == "statics":
        expand(buf, INDEX)
    return buf
''' % LOADER_DOC


def front_end_key():
    return ("%d Hz, frame %d/%d, FFT %d, %d mel %g-%g Hz, c1..c%d%s, Q%d log2"
            % (mfcc.SAMPLE_RATE, mfcc.FRAME_LEN, mfcc.FRAME_STRIDE,
               mfcc.FFT_SIZE, mfcc.N_MEL, mfcc.MEL_LOW_HZ, mfcc.MEL_HIGH_HZ,
               mfcc.N_CEPS, " + deltas D=%d" % mfcc.DELTA_WIDTH
               if mfcc.DELTA_WIDTH else "", mfcc.LOG_Q - mfcc.FEAT_SHIFT))


def build_blob(takes, pack):
    """(blob, index, n_clamped). Index entries are (label, frame_off, n)."""
    blob = bytearray()
    index = []
    frame_off = 0
    clamped = 0
    # Indexed rather than unpacked: src/vocab.py's entries have grown a field
    # once already (the noun/feeling/trigger tag) and only the first and third
    # matter here.
    for entry in vocab.VOCAB:
        label = entry[0]
        for form in entry[2]:
            for q8 in takes.get(form, []):
                if pack == "statics":
                    packed, n_clamped = mfcc.pack_statics(q8)
                    clamped += n_clamped
                else:
                    packed = mfcc.pack_template(mfcc.features_from_q8(q8))
                index.append((label, frame_off, len(q8)))
                frame_off += len(q8)
                blob += packed
    return bytes(blob), index, clamped


def emit(takes, out_dir, fmt, pack, source):
    blob, index, clamped = build_blob(takes, pack)
    total_frames = sum(n for _l, _o, n in index)
    longest = max((n for _l, _o, n in index), default=0)
    buffer_bytes = 2 * mfcc.n_feat() * total_frames
    scratch_bytes = 2 * mfcc.N_CEPS * longest if pack == "statics" else 0

    head = HEADER % (front_end_key(), time.strftime("%Y-%m-%d %H:%M"), source,
                     pack, mfcc.TEMPLATE_FORMAT, mfcc.N_CEPS, mfcc.n_feat(),
                     2 * mfcc.n_feat(), pack, total_frames, buffer_bytes,
                     len(blob), scratch_bytes)

    py = os.path.join(out_dir, "templates.py")
    body = ["INDEX = ("]
    for label, offset, n in index:
        body.append("    (%r, %d, %d)," % (label, offset, n))
    body.append(")")
    body.append("")

    if fmt == "bin":
        binpath = os.path.join(out_dir, "templates.bin")
        with open(binpath, "wb") as fh:
            fh.write(blob)
        body.append(LOADER_BIN)
        written = [binpath, py]
    else:
        body.append("BLOB = (")
        body.append(_escape(blob))
        body.append(")")
        body.append(LOADER_PY)
        written = [py]

    with open(py, "w") as fh:
        fh.write(head)
        fh.write("\n".join(body))
    return blob, index, written, clamped, buffer_bytes, scratch_bytes


def main(argv=None):
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from", dest="source_dir",
                     help="directory of WAVs, ideally tools/enrol.py's output")
    src.add_argument("--board", action="store_true",
                     help="record through the board's microphone over serial")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="where to write templates.py (and templates.bin). "
                         "The default is src/, which is where deploy.sh looks; "
                         "anywhere else is for inspection and will not deploy.")
    ap.add_argument("--format", choices=("py", "bin"), default="bin",
                    help="bin writes templates.bin plus a small loader (the "
                         "default; a .py literal of this size will not compile "
                         "on the device). py freezes the data into the module.")
    ap.add_argument("--pack", choices=("statics", "full"), default="statics",
                    help="statics stores 12-wide Q8 and the device expands to "
                         "24-wide at start-up, halving flash and transfer; "
                         "full stores the finished 24-wide features")
    ap.add_argument("--seconds", type=float, default=RECORD_SECONDS)
    ap.add_argument("--port", default=os.environ.get("PORT",
                                                     "/dev/cu.usbmodem101"))
    args = ap.parse_args(argv)

    if args.board:
        takes = takes_from_board(list(vocab.FORMS), args.seconds, args.port,
                                 mfcc.SAMPLE_RATE)
        source = "board ES8311 microphone via %s" % args.port
    else:
        takes = takes_from_dir(args.source_dir)
        source = args.source_dir

    # Enrolment is the step between adding a word and shipping it, so it is
    # the last cheap moment to catch an echo that will shrink every reply on
    # the panel. Uses vocab's declaration rather than a fresh literal.
    # vocab.MAX_ECHO_LETTERS, not a getattr fallback: a default here would be
    # a fourth copy of the ceiling, hidden in an argument, silently wrong the
    # moment the real one changed. If the constant is missing, that is a broken
    # vocab.py and enrolment should stop rather than guess.
    overlong = [(e[0], e[1]) for e in vocab.VOCAB
                if len(e[1]) > vocab.MAX_ECHO_LETTERS]

    missing = [f for f in vocab.FORMS if not takes.get(f)]
    thin = [(f, len(takes[f])) for f in vocab.FORMS
            if takes.get(f) and len(takes[f]) < REPS]

    blob, index, written, clamped, buffer_bytes, scratch_bytes = \
        emit(takes, args.out, args.format, args.pack, source)

    lens = [n for _l, _o, n in index]
    print()
    for path in written:
        print("wrote %s (%d bytes)" % (path, os.path.getsize(path)))
    print("  %d templates over %d classes, %d frames of payload"
          % (len(index), len(set(l for l, _o, _n in index)), sum(lens)))
    if lens:
        print("  template length %d..%d frames, %d mean (%d..%d ms)"
              % (min(lens), max(lens), sum(lens) // len(lens),
                 min(lens) * 10, max(lens) * 10))
    print("  payload %d bytes; device RAM to hold it: %.1f KB"
          % (len(blob), len(blob) / 1024.0))
    if missing:
        print("  MISSING, no template at all: %s" % ", ".join(missing))
    for label, echo in overlong:
        print("  ERROR: echo %r (%s) is %d letters; %d is the ceiling."
              % (echo, label, len(echo), vocab.MAX_ECHO_LETTERS))
        print("  A word this long drops EVERY reply on the panel from 16-pixel")
        print("  to 8-pixel text -- silently, and not only for this word.")
        print("  See the note in src/vocab.py.")
    if clamped:
        print("  ERROR: %d values clamped at int16 packing Q8 statics." % clamped)
        print("  The recording was loud enough to overflow the statics-only")
        print("  format, so these templates are NOT bit-exact. Re-record with")
        print("  less gain, or use --pack full (38x headroom, twice the size).")
    for form, n in thin:
        print("  thin: %s has %d of %d takes" % (form, n, REPS))
    print("  on the filesystem %d bytes; device buffer to allocate %d (%.1f KB)"
          % (len(blob), buffer_bytes, buffer_bytes / 1024.0))
    if args.pack == "statics":
        print("  statics-only: the device expands 12-wide Q8 to 24-wide at")
        print("  start-up, needing %d bytes of scratch. Halves flash and the"
              % scratch_bytes)
        print("  mpremote transfer; match-time RAM is unchanged.")
        print("  REQUIRES spotter.expand on the device -- templates.load()")
        print("  refuses without it. If the expansion is not ported yet, use")
        print("  --pack full: same RAM, twice the file, no expansion needed.")
    if args.format == "py":
        py_bytes = os.path.getsize(os.path.join(args.out, "templates.py"))
        print("  WARNING: templates.py is %d bytes of source. An escaped bytes"
              % py_bytes)
        print("  literal costs four characters per byte and MicroPython has to")
        print("  compile it inside a ~490 KB heap. Use --format bin.")
    return 1 if (missing or clamped or overlong) else 0


if __name__ == "__main__":
    sys.exit(main())
