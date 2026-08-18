#!/usr/bin/env python3
"""Does this recogniser have any margin, and does a microphone swap destroy it?

The one experiment worth running before writing anything else. It answers two
questions with numbers, in about ten minutes of talking:

1. **Same microphone: is there margin at all?** If utterances of the same word
   are not much closer together than utterances of different words, DTW over
   MFCCs is the wrong approach and no amount of threshold tuning will save it.
   This is the question that would invalidate the whole design, so it is asked
   first and asked plainly.

2. **Cross microphone: does enrolling on the Mac work?** Templates from a
   MacBook microphone matched against recognition through the board's ES8311 is
   a channel mismatch, and speaker-dependent DTW has nothing to absorb one --
   no model, three templates per word, and a cepstral mean estimated from half
   a second of audio in which the channel and the phonetics are not separable.
   The argument says it should fail. This measures how badly.

## Why it records both microphones at once

Both captures cover the same wall-clock window and hear the **same utterance**,
so the only difference between the two recordings is the channel. Recording
into each microphone in turn would confound the channel with how differently
the word was said the second time, which is the very thing being measured.

    uvx --from miniaudio --with pyserial python tools/mic_margin.py record run1/
    python3 tools/mic_margin.py report run1/          # re-analyse, no hardware
    python3 tools/mic_margin.py simulate corpus/ sim/ # dry run, no hardware
    python3 tools/mic_margin.py compare run1/         # front ends, cross-mic

`record` needs the board on USB and microphone permission for the terminal.
`report` needs neither, so the verdict can be recomputed after a front-end
change without saying MOTHER forty more times.

Every capture is written to WAV, so a session is archived and can be re-scored
against a different front end later. That matters: this experiment is cheap to
analyse and expensive to run.
"""

import array
import os
import sys
import time
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

import mfcc   # noqa: E402
import vad    # noqa: E402

if os.path.dirname(os.path.abspath(vad.__file__)) != HERE:
    raise ImportError("vad resolved to %s, wanted %s" % (vad.__file__, HERE))

# Ten words, chosen rather than sampled. MOTHER and FATHER are the tightest
# pair in the vocabulary (they share /-Vdh-uh-r/, about 60% of each word), so
# they are the pair whose margin is worth knowing. COMPUTER and CHILDREN are
# the long polysyllables that should be easy, and act as a control: if those
# collapse, something is wrong with the rig, not with the vocabulary.
WORDS = ("mother", "father", "computer", "children", "yes",
         "work", "sleep", "money", "dream", "always")
REPS = 4
SECONDS = 2.0

# From the synthetic corpus, same channel, different sessions: mean within-word
# 256, mean between-word 733, ratio 2.86. Synthetic speech is more repeatable
# than a person, so real same-microphone numbers should land below this. It is
# an upper reference point, not a target.
REFERENCE_RATIO = 2.86
RATIO_HEALTHY = 2.0
RATIO_TIGHT = 1.5


# --- Capture ---------------------------------------------------------------

class MacMic(object):
    """Background capture from the default input device, via miniaudio."""

    def __init__(self, rate):
        import miniaudio
        self.miniaudio = miniaudio
        self.rate = rate
        self.chunks = []
        self.device = None

    def _sink(self):
        while True:
            data = yield
            self.chunks.append(bytes(memoryview(data).cast("B")))

    def start(self):
        self.chunks = []
        gen = self._sink()
        next(gen)
        self.device = self.miniaudio.CaptureDevice(
            input_format=self.miniaudio.SampleFormat.SIGNED16,
            nchannels=1, sample_rate=self.rate, buffersize_msec=50)
        self.device.start(gen)

    def stop(self):
        self.device.stop()
        self.device = None
        out = array.array("h")
        out.frombytes(b"".join(self.chunks))
        if sys.byteorder != "little":
            out.byteswap()
        return out


def write_wav(path, samples, rate):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())


def record_session(outdir, port_name, seconds, rate):
    import pull_recording as pull

    os.makedirs(outdir, exist_ok=True)
    mac = MacMic(rate)
    print("Recording %d words x %d takes through BOTH microphones at once.\n"
          "Hold the board where you would actually hold it, and do not move\n"
          "between the two -- the point is that only the channel differs.\n"
          % (len(WORDS), REPS))

    with pull.open_port(port_name, timeout=0.5) as port:
        for rep in range(REPS):
            for word in WORDS:
                while True:
                    ans = input("  %-9s take %d/%d -- Enter to record, (s)kip: "
                                % (word.upper(), rep + 1, REPS)).strip().lower()
                    if ans.startswith("s"):
                        break
                    mac.start()
                    time.sleep(0.15)   # let the input device settle first
                    print("    speak now")
                    try:
                        got, pcm, elapsed = pull.capture(port, seconds=seconds,
                                                         rate=rate)
                    except pull.CaptureError as exc:
                        mac.stop()
                        print("    board capture failed: %s" % exc)
                        continue
                    mac_samples = mac.stop()

                    board = array.array("h")
                    board.frombytes(pcm)
                    if sys.byteorder != "little":
                        board.byteswap()

                    ok_b = vad.endpoints(board)
                    ok_m = vad.endpoints(mac_samples)
                    if ok_b is None or ok_m is None:
                        print("    no utterance found on %s -- again"
                              % ("the board" if ok_b is None else "the Mac"))
                        continue
                    write_wav(os.path.join(outdir, "board_%s_%d.wav"
                                           % (word, rep)), board, rate)
                    write_wav(os.path.join(outdir, "mac_%s_%d.wav"
                                           % (word, rep)), mac_samples, rate)
                    print("    board %.0f ms, mac %.0f ms, %.2f s"
                          % ((ok_b[1] - ok_b[0]) / (rate / 1000.0),
                             (ok_m[1] - ok_m[0]) / (rate / 1000.0), elapsed))
                    break
    print("\nwrote %s" % outdir)


# --- Analysis --------------------------------------------------------------

def load(outdir):
    """{mic: {word: [frames, ...]}}, endpointed and turned into features."""
    out = {"board": {}, "mac": {}}
    missing = []
    for fn in sorted(os.listdir(outdir)):
        if not fn.endswith(".wav") or "_" not in fn:
            continue
        mic, rest = fn.split("_", 1)
        if mic not in out:
            continue
        word = rest.rsplit("_", 1)[0]
        samples = vad.read_wav(os.path.join(outdir, fn))
        trimmed = vad.trim(samples)
        if trimmed is None:
            missing.append(fn)
            continue
        out[mic].setdefault(word, []).append(mfcc.mfcc(trimmed))
    return out, missing


def condition(tmpl_set, query_set):
    """(mean within, mean between, ratio, n_within, n_between).

    A take is never compared against itself: a template and a query drawn from
    the same recording would score ~0 and flatter the within-word mean into
    meaninglessness. For the same-microphone conditions that means take i of a
    word is scored only against takes j != i.
    """
    import dtw as dtw_mod
    within, between = [], []
    same = tmpl_set is query_set
    for word, queries in sorted(query_set.items()):
        for qi, q in enumerate(queries):
            for other, tmpls in sorted(tmpl_set.items()):
                for ti, t in enumerate(tmpls):
                    if same and other == word and ti == qi:
                        continue
                    d = dtw_mod.dtw(q, t)
                    if d >= dtw_mod.INF:
                        continue
                    (within if other == word else between).append(d)
    if not within or not between:
        return None
    mw = sum(within) / len(within)
    mb = sum(between) / len(between)
    return mw, mb, mb / mw, len(within), len(between)


def top1(tmpl_set, query_set):
    """Fraction of queries whose nearest template is the right word.

    The ratio says whether the geometry is right; this says whether the thing
    would actually work. They can disagree -- a healthy mean ratio with a long
    tail still misfires.
    """
    import dtw as dtw_mod
    hits = total = 0
    same = tmpl_set is query_set
    for word, queries in sorted(query_set.items()):
        for qi, q in enumerate(queries):
            best, best_word = dtw_mod.INF, None
            for other, tmpls in sorted(tmpl_set.items()):
                for ti, t in enumerate(tmpls):
                    if same and other == word and ti == qi:
                        continue
                    d = dtw_mod.dtw(q, t)
                    if d < best:
                        best, best_word = d, other
            if best_word is not None:
                total += 1
                hits += (best_word == word)
    return hits, total


def verdict(board, mac, cross):
    """The point of the whole tool: one answer, in words."""
    lines = []
    if board is None:
        return ["INCONCLUSIVE: not enough board audio to score."]

    b_ratio = board[2]
    lines.append("")
    if b_ratio < RATIO_TIGHT:
        lines.append("VERDICT 1/2  DTW HAS NO MARGIN ON ITS OWN MICROPHONE.")
        lines.append("  Same-word and different-word distances overlap"
                     " (ratio %.2f, want >= %.1f)." % (b_ratio, RATIO_HEALTHY))
        lines.append("  This is not a threshold problem and tuning will not fix"
                     " it. Stop and")
        lines.append("  re-examine the approach before building anything on"
                     " top of it.")
    elif b_ratio < RATIO_HEALTHY:
        lines.append("VERDICT 1/2  Same-microphone margin is TIGHT"
                     " (ratio %.2f)." % b_ratio)
        lines.append("  Workable, but expect the rejection threshold to buy"
                     " precision at a")
        lines.append("  steep cost in recall. More templates per word is the"
                     " cheapest lever.")
    else:
        lines.append("VERDICT 1/2  Same-microphone margin is HEALTHY"
                     " (ratio %.2f)." % b_ratio)
        lines.append("  Reference from the synthetic corpus is %.2f, and"
                     " synthetic speech is" % REFERENCE_RATIO)
        lines.append("  more repeatable than a person, so at or below that is"
                     " expected.")

    lines.append("")
    if cross is None or mac is None:
        lines.append("VERDICT 2/2  INCONCLUSIVE: no paired Mac audio to"
                     " compare against.")
        return lines

    c_ratio = cross[2]
    keep = c_ratio / b_ratio if b_ratio else 0.0
    lines.append("VERDICT 2/2  Cross-microphone ratio %.2f vs %.2f"
                 " same-microphone (%.0f%% kept)."
                 % (c_ratio, b_ratio, 100 * keep))
    if c_ratio < RATIO_TIGHT or keep < 0.6:
        lines.append("  MAC-RECORDED TEMPLATES ARE NOT VIABLE. Enrol through"
                     " the board.")
        lines.append("  The margin that survives the channel swap is too small"
                     " to set a")
        lines.append("  rejection threshold in -- the wrong word and the right"
                     " word are")
        lines.append("  about equally far away, so mismatch produces confident"
                     " errors")
        lines.append("  rather than silence.")
    elif keep < 0.85:
        lines.append("  Mac templates cost real margin. Usable for development"
                     " scaffolding,")
        lines.append("  not for anything whose accuracy is being quoted."
                     " Enrol on the board.")
    else:
        lines.append("  The channel swap costs little here. Worth re-checking"
                     " in a different")
        lines.append("  room before relying on it -- one room is one channel.")
    return lines


def report(outdir):
    sets, missing = load(outdir)
    board, mac = sets["board"], sets["mac"]
    n_b = sum(len(v) for v in board.values())
    n_m = sum(len(v) for v in mac.values())
    print("front end: %d Hz, %d mel %g-%g Hz, c1..c%d%s, %d features/frame"
          % (mfcc.SAMPLE_RATE, mfcc.N_MEL, mfcc.MEL_LOW_HZ, mfcc.MEL_HIGH_HZ,
             mfcc.N_CEPS, " + deltas" if mfcc.DELTA_WIDTH else "",
             mfcc.n_feat()))
    print("%s: %d board and %d Mac utterances over %d words"
          % (outdir, n_b, n_m, len(board)))
    for fn in missing:
        print("  endpointing found nothing in %s" % fn)
    if not n_b:
        print("\nno board audio -- run `record` first")
        return 1

    conds = []
    b = condition(board, board)
    conds.append(("board -> board", b, top1(board, board)))
    m = condition(mac, mac) if n_m else None
    if m:
        conds.append(("mac -> mac", m, top1(mac, mac)))
    cross = condition(mac, board) if n_m else None
    if cross:
        conds.append(("mac templates -> board queries", cross,
                      top1(mac, board)))

    print("\n%-32s %8s %8s %7s %9s"
          % ("condition", "within", "between", "ratio", "top-1"))
    for name, c, (hits, total) in conds:
        if c is None:
            continue
        mw, mb, ratio, _nw, _nb = c
        print("%-32s %8.0f %8.0f %7.2f %6d/%-3d"
              % (name, mw, mb, ratio, hits, total))

    for line in verdict(b, m, cross):
        print(line)
    return 0


def simulate(corpus, outdir):
    """Build a paired session by passing corpus audio through two channels.

    Not a substitute for the real thing -- the "microphone difference" is one I
    chose, so this predicts rather than measures. It exists so that the
    analysis half can be proven correct before anyone spends ten minutes
    talking to a board, and so the real run has a number to be compared
    against rather than being the first data point ever seen.
    """
    import say_corpus as sc

    os.makedirs(outdir, exist_ok=True)
    rng_b, rng_m = sc.Rng(11), sc.Rng(22)
    made = 0
    for word in WORDS:
        rep = 0
        for sub in ("enrol", "test"):
            d = os.path.join(corpus, sub)
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                if not fn.startswith(word + "."):
                    continue
                clean = sc.read_wav(os.path.join(d, fn))
                # "board": the channel the corpus was already built for.
                board = sc.channel(clean, rng_b, -40, 0, 0.0)
                # "mac": brighter, quieter floor, more gain. A plausible
                # difference between a laptop array mic and a small board
                # electret, and the hypothesis under test.
                mac = sc.channel(clean, rng_m, -52, 6, 0.35)
                write_wav(os.path.join(outdir, "board_%s_%d.wav" % (word, rep)),
                          board, sc.RATE)
                write_wav(os.path.join(outdir, "mac_%s_%d.wav" % (word, rep)),
                          mac, sc.RATE)
                rep += 1
                made += 1
    print("simulated %d paired utterances in %s" % (made, outdir))
    print("(the channel difference is chosen, not measured -- this is a"
          " prediction)\n")


# Front ends worth comparing. The same-microphone corpus cannot separate these
# -- every one of them scores 0.990 there, because they are channel-robustness
# measures and that corpus has no channel mismatch in it. Cross-microphone is
# the condition where they either earn their cost or do not.
VARIANTS = (
    ("baseline 100-7600", {}),
    ("deltas D=2", {"DELTA_WIDTH": 2}),
    ("mel 300-3400 x20", {"MEL_LOW_HZ": 300.0, "MEL_HIGH_HZ": 3400.0,
                          "N_MEL": 20}),
    ("mel 300-3400 + deltas", {"MEL_LOW_HZ": 300.0, "MEL_HIGH_HZ": 3400.0,
                               "N_MEL": 20, "DELTA_WIDTH": 2}),
    ("mel 200-5000 x24", {"MEL_LOW_HZ": 200.0, "MEL_HIGH_HZ": 5000.0,
                          "N_MEL": 24}),
    ("mel 200-5000 + deltas", {"MEL_LOW_HZ": 200.0, "MEL_HIGH_HZ": 5000.0,
                               "N_MEL": 24, "DELTA_WIDTH": 2}),
)

_TUNABLE = ("DELTA_WIDTH", "MEL_LOW_HZ", "MEL_HIGH_HZ", "N_MEL")


def compare(outdir):
    """Re-score one recorded session under several front ends.

    This is the experiment that actually decides whether deltas and a
    band-limited filterbank are worth their cost, and it is why the session is
    archived as WAVs rather than as features. Run it on the real recording the
    moment there is one.
    """
    saved = dict((k, getattr(mfcc, k)) for k in _TUNABLE)
    print("%-24s %7s %7s %7s   %8s %7s"
          % ("front end", "same", "cross", "kept", "cross", "bytes/"))
    print("%-24s %7s %7s %7s   %8s %7s"
          % ("", "ratio", "ratio", "", "top-1", "frame"))
    rows = []
    for name, over in VARIANTS:
        for k, v in saved.items():
            setattr(mfcc, k, v)
        for k, v in over.items():
            setattr(mfcc, k, v)
        mfcc._TABLES = None
        sets, _missing = load(outdir)
        board, mac = sets["board"], sets["mac"]
        if not board or not mac:
            print("%-24s  (not enough paired audio)" % name)
            continue
        b = condition(board, board)
        c = condition(mac, board)
        hits, total = top1(mac, board)
        keep = c[2] / b[2] if b and b[2] else 0.0
        rows.append((name, b[2], c[2], keep, hits, total))
        print("%-24s %7.2f %7.2f %6.0f%%   %5d/%-3d %7d"
              % (name, b[2], c[2], 100 * keep, hits, total,
                 2 * mfcc.n_feat()))
    for k, v in saved.items():
        setattr(mfcc, k, v)
    mfcc._TABLES = None
    if rows:
        best = max(rows, key=lambda r: (r[4], r[3]))
        print("\nbest cross-microphone: %s (%d/%d top-1, %.0f%% of margin"
              " kept)" % (best[0], best[4], best[5], 100 * best[3]))
    return 0


def main(argv):
    if len(argv) >= 3 and argv[1] == "record":
        port = os.environ.get("PORT", "/dev/cu.usbmodem101")
        for i, a in enumerate(argv):
            if a == "--port":
                port = argv[i + 1]
        try:
            record_session(argv[2], port, SECONDS, mfcc.SAMPLE_RATE)
        except ImportError as exc:
            print("missing a package: %s\n"
                  "  uvx --from miniaudio --with pyserial python %s"
                  % (exc, " ".join(argv)), file=sys.stderr)
            return 1
        return report(argv[2])
    if len(argv) >= 3 and argv[1] == "report":
        return report(argv[2])
    if len(argv) >= 3 and argv[1] == "compare":
        return compare(argv[2])
    if len(argv) >= 4 and argv[1] == "simulate":
        simulate(argv[2], argv[3])
        return report(argv[3])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
