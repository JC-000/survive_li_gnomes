# Speaking the reply — voice and corpus design

Giving DOCTOR a mouth. The ELIZA program currently *displays* its reply on the
e-paper; this document is about speaking it instead, in the warm, low, unhurried
delivery the user asked for.

It records what was established on 2026-08-18 about which macOS voice to use,
what prosody `say` will actually honour, and — the consequential part — how much
of the reply corpus fits in 3 MB once the voice is chosen.

Companion to [speech-design.md](speech-design.md), which covers the input side
(why the ball spots keywords rather than transcribing). This is the output side.
The same four labels are used, and mean the same things:

| Label | Means |
| --- | --- |
| *measured* | Measured in this project — on the **host** unless it says otherwise |
| *published* | Someone else measured it, on hardware named in the citation |
| *estimate* | Arithmetic from a *published* anchor, which is always named |
| *unknown* | Not established either way |

Everything here is *measured* on the Mac unless marked. Nothing in this document
has been played through the board.

## The one thing this document cannot tell you

**Nothing here was arrived at by listening.** The audition was built, measured
and reasoned about by an agent that cannot hear audio. Every ranking below is
derived from numbers — fundamental frequency, duration, pitch discontinuity —
plus stated reasoning from them, and each one names its criterion so it can be
disagreed with.

That is a real limit and not a modest one. Timbre, breathiness, apparent age and
whether a voice is *appealing* do not appear in any measurement taken here. The
user has already demonstrated the gap by rejecting Noelle — "sounds like a female
child" — which no number in this file predicts.

So: **the audition is the deliverable and the ear is the instrument.** The
recommendation below is a starting position, not a verdict.

## Correction, 2026-08-19: the corpus is 113 clips and 1.89 MB

Everything below about voice, prosody and the seam stands. **The sizing does
not.** The figures in "The short version", "The seam" and "What fits" are built
on a corpus of 379 sentences, and that count is wrong for this device by a
factor of three and a bit.

379 is every reply template in `eliza_rules` crossed with every filler it
grammatically accepts. The device cannot reach most of them.
`talk.Conversation.reply` hands the engine a bag of **exactly one** spotted
word, and `Doctor._answer` returns the *first* keyword rule that produces a
reply -- so a bag of {MOTHER} is answered by the family rules and the twenty-odd
other templates that would also accept "mother" are never consulted. That is a
property of `talk.py`, not of the script, and it is the same fact
`tools/test_talk.py` already relies on to call PHRASE and MEMORY unreachable.

Enumerated over the rule data taking every branch and every rotation state --
`tools/voice_pak.py::corpus()`, checked against the engine itself by
`tools/test_voice_pak.py` -- the device-reachable corpus is:

| | Clips | Seconds | 16 kHz int16 | 16 kHz IMA ADPCM |
| --- | --- | --- | --- | --- |
| device-reachable replies | 111 | 244 | 7.45 MB | 1.86 MB |
| + greeting, + NOTHING\_HEARD | **113** | **248** | **7.56 MB** | **1.89 MB** |

*(measured -- every clip rendered in Ava (Premium) at p3-warm, `say` padding
trimmed, peak-normalised to 15000, encoded and round-tripped)*

Against the reflashed board's 15 MB filesystem that is **12.6% used, 13.1 MB
spare**, and the uncompressed corpus would have fitted too. So:

- **The seam question is moot.** Splicing existed to turn 379 clips into 132.
  There is nothing to save. Every reply is rendered whole, which is the reading
  the measurement preferred anyway. The seam work is kept below because it is
  the answer to "can `say` fragments be joined" and that question will come
  back.
- **The 8 kHz-versus-vocabulary trade is moot.** Nothing has to be dropped and
  nothing has to be halved. 16 kHz, full vocabulary, all twelve nouns.
- **The 3 MB filesystem the whole budget was reasoned against is gone.** The
  board was reflashed to 16 MB during the same session (firmware `e7b1069a`),
  which is what makes the margin as wide as it is. Even at the old 3 MB the
  ADPCM corpus would have fitted with 1.1 MB to spare.

Of the 111, only **68 were observed** in a 30,000-turn interleaved walk over the
real `talk.Conversation`; the other 43 need a rule-rotation state the walk never
produced but that nothing in the engine forbids. All 113 ship. A clip nobody
plays costs ~17 KB of 15 MB; a missing clip is a reply the toy mouths silently,
and at the desk that is indistinguishable from a decoder bug or a flat battery.

## The short version

1. **Speaking every reply does not fit, at any audio format.** The full corpus is
   379 sentences, 885 s, and **3.38 MB even at 8 kHz 4-bit ADPCM** — over a 3 MB
   filesystem that also holds the code, the templates and the existing clips.
   This is the finding that constrains the feature.
2. **Splicing makes it fit easily** — 132 clips, 249 s, **1.90 MB at 16 kHz**,
   over a megabyte spare. Splicing is therefore not an optimisation. It is the
   difference between speaking the whole script and not.
3. **But the splice measures badly**: a pitch step of most of an octave at every
   join, where the intact sentence moves by a tone, and no `say` control can
   smooth it. Whether that is disqualifying is an ear judgement nobody has made
   yet, and it is the single most consequential thing still open.
4. **If it is disqualifying, the corpus truncates gracefully.** A template with
   no clip for the noun it heard falls through to a deflection, which is a real
   DOCTOR response. Roughly 7 nouns fit at 8 kHz, 2 at 16 kHz.
5. **Recommended voice: Ava (Premium) at `p3-warm`** — `[[pbas 42]]`, `-r 100`,
   one `[[slnc 220]]` — landing at 148 Hz from a 172 Hz natural.

## Why pre-rendered, and why that makes vocabulary the constraint

The board cannot synthesise speech at runtime. The alternatives were costed
before this work began:

- **SAM**, the 1982 Commodore 64 engine, has a MicroPython port. Pure Python
  renders "Hello World" in ~1900 ms; only its native C `.mpy` reaches ~17 ms, and
  the shipped build is `armv6m` so it would need rebuilding for this board's
  `armv7emsp`. Ruled out as not worth the toolchain. *(published)*
- **espeak-ng** is ~1.46 MB of program storage and expects PSRAM this board does
  not have. Ruled out. *(published)*

So every line the device will ever speak is rendered here, on the Mac, at build
time, and shipped as audio. That is why "which voice, at what prosody" is a
decision taken once and frozen — and why the binding question stops being
*quality* and becomes *how many distinct sentences fit in 3 MB*.

## What `say` actually honours

Not what the documentation implies, and the difference matters because two of
the obvious levers are gone.

Each row was established by rendering the same line twice at different settings
and comparing MD5s. **An ignored embedded command leaves the output
byte-identical**, and that is the only way to tell it apart from one that worked
— a setting that renders successfully and changes nothing is otherwise
indistinguishable from a setting you have tuned badly.

| Command | Status |
| --- | --- |
| `[[pbas n]]` pitch base | **Works.** 44 is neutral for every voice tried; ~1.25 semitones per unit; clamps around 34 |
| `[[rate n]]`, `-r` | **Works**, but see below |
| `[[volm f]]` volume | **Works** |
| `[[slnc ms]]` silence | **Works.** The only control that buys an arbitrary pause |
| `[[pmod n]]` pitch modulation | **Ignored** |
| `[[emph +]]` emphasis | **Ignored** |
| `[[inpt PHON]]` phonemes | Parsed and stripped, then the phoneme string is read out as letters |

`pmod` is ignored on `Allison (Enhanced)`, on `Samantha`, **and on `Fred`** — so
this is not a modern-voice regression, it is gone across the board including the
1990s formant synthesiser. That removes the obvious lever for a flatter, warmer
line reading, and it removes the only tool that could have smoothed a splice
after the fact. It matters again in [The seam](#the-seam).

### `-r` barely slows these voices down

The single most useful thing in this file for anyone trying to make a voice sound
unhurried. Allison (Enhanced), "Tell me more about your family":

| `-r` | Duration | | `-r` | Duration |
| --- | --- | --- | --- | --- |
| 300 | 1.01 s | | 140 | 1.70 s |
| 220 | 1.32 s | | 100 | 1.89 s |
| 160 (default) | 1.61 s | | 50 | 2.14 s |

Asking for a **3.2× slowdown delivers 1.33×**, and adjacent values quantise to
byte-identical output (160 and 180 are the same file; so are 100 and 80).
Speeding *up* tracks the request fine — the compression is only on the slow side.

**A languid delivery therefore cannot be bought with `-r`.** It has to come from
inserted silence, and `[[slnc]]` is the only control that provides any.

Punctuation is not a substitute: a comma, an ellipsis and an em dash produce
**byte-identical** output, a fixed ~340 ms break with no way to lengthen it.

### Rate and pitch are not independent

Worth knowing before reading a `pbas` number as a pitch. Allison, same line,
median F0 by autocorrelation:

| Setting | Median F0 |
| --- | --- |
| default | 200.5 Hz |
| `-r 100` alone | 193.4 Hz |
| `[[pbas 42]]` alone | 172.3 Hz |
| `[[pbas 42]]` with `-r 100` | 148.0 Hz |

Slowing down lowers pitch on its own, and the two compound more than they add
(172 × 193/200 predicts 166, not 148). So presets are labelled with both the
`pbas` they ask for and the F0 they land on, because those are different numbers.

## The voices

All eleven the user asked for are installed as of 2026-08-18. Natural pitch, no
markup, on "Tell me more about your family":

| Voice | Tier | F0 | |
| --- | --- | --- | --- |
| Susan | Enhanced | 162.1 Hz | Siri-generation |
| Nicky | Enhanced | 165.8 Hz | |
| Samantha | compact | 170.9 Hz | |
| **Ava** | **Premium** | **172.3 Hz** | |
| Samantha | Enhanced | 180.7 Hz | |
| Allison | Enhanced | 200.5 Hz | |
| Zoe | Premium | 220.5 Hz | Siri-generation |
| Joelle | Enhanced | 237.1 Hz | |
| Noelle | Enhanced | 268.9 Hz | **excluded, see below** |
| Tom | Enhanced | 128.9 Hz | male |
| Evan | Enhanced | 121.2 Hz | male |
| Nathan | Enhanced | 107.0 Hz | male |

### `say -v` substitutes the default voice silently, and it cost real time here

An uninstalled voice does not raise, warn, or exit non-zero. `say` renders the
line in the system default and reports success.

This fired hard during the work: at the start of the session, nine of the eleven
requested names rendered **byte-identical** files — `d1f850c3…`, 78332 bytes,
all of them Samantha. A shortlist built without checking would have had the user
auditioning one voice nine times under nine labels, forming opinions about
differences that did not exist, and there would have been nothing in the output
to suggest anything was wrong.

`tools/voice_audition.py` refuses to run if any two shortlisted voices hash the
same. This is the same trap as the corpus one already recorded in
`.serena/memories/gotchas_that_cost_time.md`; it is worth knowing that it also
catches *partially written assets*, since a voice appears in `say -v '?'` before
its data has finished downloading.

**Always include the quality suffix.** `Samantha` and `Samantha (Enhanced)` are
both installed and are *different voices* — 170.9 vs 180.7 Hz, different MD5s.
The bare name gets you the compact tier, which is precisely the tier this
exercise exists to avoid. Where only one tier exists the bare name resolves to it
(`Zoe` and `Zoe (Premium)` are byte-identical), so the suffix is harmless to
include and dangerous to omit.

### Audition for apparent age, not only for warmth

The user rejected **Noelle** by ear: *"sounds like a female child."* That is a
judgement about character rather than quality, no measurement here predicts it,
and it is decisive.

It also gives the shortlist an axis it was missing. Noelle is the
highest-pitched voice installed at 269 Hz; Joelle (237) and Zoe (220) are the
next two. **That is a hypothesis for the ear, not a finding** — apparent age is
not a function of F0, and a voice can read young for reasons of timbre and
articulation that pitch says nothing about. So both stay in the shortlist,
ordered by pitch and flagged, rather than being quietly dropped on a
correlation. If they also read young, the correlation is worth trusting next
time; if they do not, it should be discarded.

## The audition

Built by `tools/voice_audition.py`, output in `corpus-voice/`.

    uv run tools/voice_audition.py corpus-voice/            # shortlist, ~1 min
    uv run tools/voice_audition.py corpus-voice/ --budget   # + sizing, ~6 min

**`corpus-voice/` is gitignored** (by the existing `corpus*/` rule), so the audio
does not survive a fresh clone. The tool regenerates it exactly; the settings
live in `PRESETS` and `SHORTLIST` in that file and nowhere else.

Each `__REEL.wav` is the same five real DOCTOR lines back to back, 9–14 s. Reels
first — they are much the fastest way to form an opinion, and the per-line files
are there only for a closer listen afterwards.

    cd corpus-voice
    afplay audition/ava-premium__p0-neutral__REEL.wav    # 1. the control
    afplay audition/ava-premium__p3-warm__REEL.wav       # 2. the recommendation
    afplay audition/ava-premium__p4-darker__REEL.wav     # 3. one step too far?
    for f in audition/*__p3-warm__REEL.wav; do afplay "$f"; done   # 4. the voices

`INDEX.md` in that directory names every file, its settings and its measured F0.

### The presets

| | `pbas` | `-r` | Pause | What it is for |
| --- | --- | --- | --- | --- |
| p0-neutral | — | — | — | control: no markup at all |
| p1-lower | 42 | — | — | pitch alone |
| p2-slow | — | 100 | — | rate alone — note how little it does |
| p3-light | 43 | 100 | 220 ms | half the pitch drop |
| **p3-warm** | **42** | **100** | **220 ms** | **the recommendation** |
| p4-darker | 40 | 100 | 260 ms | −5 st, probably past the point |
| p5-intimate | 41 | 60 | 320 ms | everything, plus `volm 0.75` and a lead-in |

The pause is one `[[slnc]]` inserted before the last function word with two words
either side of it — which on DOCTOR's short one-clause replies lands on "about
your family", "of your mother", "when you think". A rule, not a prosodic
analysis; it is `BREAK_BEFORE` in the tool and it is the first thing to adjust if
the phrasing sounds wrong.

### Recommendation

**Ava (Premium) at p3-warm**, landing at 148 Hz. Stated criteria, in order:

1. It is one of only two Premium-tier voices installed, a tier above everything
   else on the list.
2. At 172 Hz natural it already sits in the lower half of the range. This matters
   because `[[pbas]]` shifts pitch **without moving the formants with it** — the
   further a voice is dragged, the less it sounds like a woman speaking low and
   the more like a recording slowed down. A voice that starts near the target
   needs less of the control that does the damage.

Susan (162 Hz) and Nicky (166 Hz) start lower still and are the ones to try if
Ava is dragged too far; they are Enhanced rather than Premium. `p4-darker` is in
the shortlist specifically as the point where the formant problem should become
audible — if it does not, there is more room than assumed.

## The seam

The comparison that decides the storage budget, and the one result nobody can
reconstruct from this document without redoing the work.

A slotted template like "WHY DO YOU REMEMBER _ JUST NOW" can either be **rendered
whole** once per noun (12 clips) or **assembled** from a stem and a separately
rendered noun (1 clip plus a shared noun). Across the corpus that is 379 clips
versus 132 — the difference between not fitting and fitting comfortably.

Normally you would flatten the fragments to match each other with `[[pmod 0]]`.
That is exactly the control that turns out to be ignored. **What is rendered is
what ships.**

### What the measurement says

Pitch either side of each join, `p3-warm`, Ava (Premium), against the same
measurement taken on the intact sentence at the same point:

| | Joins | Step, assembled | Step, whole |
| --- | --- | --- | --- |
| `medial-thinking` | 2 | 0.0, 7.6 st | 1.1, 1.6 st |
| `medial-remember` | 2 | 10.3, 8.9 st | 3.3, 0.0 st |
| `trailing-say` | 1 | 10.6 st | 0.2 st |
| `trailing-comes` | 1 | 10.2 st | 0.3 st |

A 0.0 means one side was unvoiced — a stop or a silence — where there is no pitch
to be discontinuous, and those are the joins that tend to survive.

The raw numbers show why, and they are physically coherent rather than an
artefact of the estimator:

    head fragment ends at    89-101 Hz     say gives every fragment a terminal
    next fragment starts at 156-168 Hz     falling contour, then starts fresh
    intact sentence, same point 120-154 Hz on both sides

**`say` imposes a complete-utterance contour on whatever string it is handed.**
The stem falls to a full stop before the noun arrives, and the noun then begins
as though starting a new sentence. The join is a ~75 Hz upward step — most of an
octave — where the intact sentence steps by 0.2–3.3 semitones.

Punctuation does not rescue it. Trailing comma, hyphen, ellipsis and a trailing
"and" were all tried on the stem, and leading comma and "and" on the filler:
every stem still ends at 88–96 Hz and every filler still starts at 152–165 Hz.

### Verdict, and its limits

**Measurement says the seam is large: about an octave, where the intact sentence
moves by a tone.** It is not a subtle artefact and it is not fixable with any
control `say` exposes.

**But acceptability is not a measurement, and I could not listen.** A toy that
already plays a fart may carry an audible splice perfectly well — the illusion
this program protects is conversational, not acoustic. Play these before
deciding:

    afplay audition/seam__medial-remember__ASSEMBLED.wav
    afplay audition/seam__medial-remember__WHOLE.wav

### The obvious mitigation does not work

The natural response to "five templates have a mid-sentence slot" is to expand
those five whole and assemble only the sentence-final ones, so the seam has to
survive in the easy case only. That was the expectation this test was built to
confirm, and **the measurement refuses it.**

Read the table again by row rather than by column. The trailing-slot pair step by
**10.6 and 10.2 semitones** — the two *largest* figures measured. The medial pair
step by 10.3/8.9 and 0.0/7.6. The easy case is not easier.

The reason follows from the mechanism: the damage is done by the head fragment
falling to a terminal contour, and a head fragment does that whether the slot
after it is mid-sentence or final. A medial slot adds a *second* join; it does
not make the first one worse. So the exception buys one join out of two on five
templates and does nothing at all for the other twenty-three.

There is no cheap subset of the corpus where splicing sounds acceptable and no
cheap variant of the technique that fixes it. The only fix that would work is
carrier-sentence rendering — render each stem inside a complete sentence so it
gets a non-final contour, then cut the filler out at a precise sample boundary.
That needs forced alignment or 33 stems edited by hand, and it was not attempted.
It is the first thing to try if the splice is wanted badly enough.

### So the two halves of the budget are in tension

This is the central unresolved question and it should not be smoothed over in
either direction:

- **Whole-sentence rendering sounds right and does not fit.** Not at any format:
  3.38 MB at 8 kHz 4-bit ADPCM against a 3 MB filesystem that also holds the
  code, the DTW templates and the existing clips.
- **Splicing fits with room to spare** — 132 clips, 249 s, **1.90 MB at 16 kHz**,
  over a megabyte clear — **and measures badly.**

If the seam turns out to be acceptable by ear, splicing is the answer and the
curve below is moot. If it does not, nothing rescues the full corpus and the
curve is the answer: ship fewer nouns and let the rest deflect.

## What fits

Measured on the full corpus rendered in Ava (Premium) at p3-warm, `say` padding
trimmed. Sizes are 4-bit IMA ADPCM, the only format anything fits in.

**The full corpus does not fit at any format** — 3.38 MB at 8 kHz and 6.76 MB at
16 kHz, against a 3 MB filesystem that also has to hold the code, the DTW
templates and the existing shake/fart/laugh clips.

It does not have to. A template with no clip for the noun it heard can **fall
through to a deflection**, exactly as an unrecognised word already does — and per
[speech-design.md](speech-design.md), a deflection is a real DOCTOR response
rather than a failure. So the corpus can be truncated anywhere on this curve and
the program still works. It just gets shyer, which this project has already
decided is the good failure.

| Vocabulary | Clips | Seconds | 16 kHz | 8 kHz |
| --- | --- | --- | --- | --- |
| canned replies only | 83 | 179 | 1.37 MB | 0.68 MB |
| + the 4 feelings | 115 | 271 | 2.07 MB | 1.03 MB |
| + 2 nouns | 159 | 373 | 2.85 MB | 1.42 MB |
| + 4 nouns | 203 | 477 | 3.64 MB | 1.82 MB |
| + 7 nouns | 269 | 636 | 4.85 MB | 2.42 MB |
| + 10 nouns | 335 | 786 | 6.00 MB | 3.00 MB |
| + 12 nouns (all) | 379 | 885 | 6.76 MB | 3.38 MB |

Leaving ~1 MB for everything else, the practical stop is around **7 nouns at
8 kHz** or **2 at 16 kHz**. That is the real choice: a wider vocabulary at
telephone quality, or a narrower one that sounds better. Given that the whole
point of this exercise was a voice worth listening to, the narrower-and-better
end deserves the benefit of the doubt.

`src/es8311.py` already carries the coefficient row for 8 kHz, so the codec side
of dropping the rate is a constant change *(unverified — not driven on the
board)*.

### Template accounting

From `src/eliza_rules.py`, which is generated and authoritative. 211 entries, of
which 191 carry reply text:

| Kind | Count | Needs |
| --- | --- | --- |
| CANNED | 79 | nothing — always speakable |
| LITERAL | 12 | a word the decomposition pinned down |
| NOUN | 16 | a noun the spotter heard |
| PHRASE | 84 | a clause the device never had |

So **107 of 191 are reachable** from keyword spotting, and 28 of those need a
word echoed into a slot. Fully expanded against the 12 spottable nouns and 4
feelings, plus the four deflections and the greeting, that is 379 sentences.

An earlier brief circulated the figures "79 canned, 21 echo-a-word"; the 21 is
stale and the correct split is 12 LITERAL + 16 NOUN = 28.

## Still open

- **Nothing has been played through the board's speaker.** Per `CLAUDE.md`, a
  codec that ignored a register write looks exactly like one that accepted it.
  The 8 kHz reconfiguration in particular is *unknown* and must be confirmed
  electrically, not by absence of an exception.
- **The seam verdict needs a human.** The measurement is unambiguous about the
  size of the discontinuity and says nothing about whether it matters.
- **Apparent age was not measured and cannot be.** The Noelle exclusion is the
  only datapoint; the F0 ordering is a guess at a proxy.
- **IMA ADPCM decode cost on this chip is *unknown*.** The sizing above assumes
  4-bit ADPCM throughout, which needs a `@micropython.viper` decoder fast enough
  to keep the codec fed. `speech-design.md` flags viper throughput on this chip
  as unmeasured, and this depends on the same missing number.
- **Whether a seam can be avoided by carrier-sentence rendering is *unknown*.**
  Rendering the stem inside a full sentence and cutting the noun out would give
  it a non-final contour, which is the actual fix. It needs a cut at a precise
  sample boundary — forced alignment, or 33 stems edited by hand — and was not
  attempted.

## The 16 kHz corpus (built 2026-08-19)

The playback path is **proven at the desk** — the user judged the repaired
8 kHz stopgap "acceptable" — so this job is an upgrade with no unknowns in
front of it, not a rescue.

What exists and is confirmed working on hardware:
- `listen.speak(i2c, path)`: re-clocks codec+MCLK around a clip, plays through
  the shared capture buffer, restores 16 kHz capture, never raises. Volume for
  speech is 82 (90 overdrives; bench-measured twice).
- `audio_pio_mpy.dma_play_words_async(buf, count=words)` — the count override
  (without it, an int16 buffer plays double: the static-burst bug).
- `talk._clip_for(text)`: `say_<sha1(text)[:8]>.pcmw` looked up per reply;
  no clip = silent panel-only reply, the correct degradation.
- Six 8 kHz stopgap clips are deployed; replace them.

The job, and where each part stands:

1. **Render every device-reachable template at 16 kHz.** **Done.**
   `tools/voice_pak.py`, on the Ava (Premium) / p3-warm recipe imported from
   `tools/voice_audition.py` rather than restated, peak 15000, `say` padding
   trimmed. 113 clips, 248 s. Ids are `sha1(text)[:8]` and
   `tools/test_voice_pak.py` proves they are what `_clip_for` looks up by
   *running* `_clip_for` against files named by the tool, rather than by
   comparing two strings.

   The mood tags are recorded per clip in `voice_manifest.txt` and are **not**
   acted on: the audition settled on one prosody for everything and a second
   recipe is a second thing to A/B at the bench. ECHO replies get the shorter
   rise for free anyway, because they are two words long and the `[[slnc]]`
   phrasing rule needs two words either side of the break to fire.

2. **Stream from flash instead of load-whole-clip.** Still required, and the
   correction above does not change that: the longest clip is
   "Don't you believe that dream has something to do with your problem?" at
   **4.48 s**, and the 96 KB shared buffer holds 1.54 s of 16 kHz packed
   stereo words. Three times over, not marginal. *(measured)*

3. **Storage.** Settled, and the premise was wrong twice over: the corpus is
   113 clips rather than 379, and the "~57 MB packed" figure counted the 4x
   expansion to stereo PIO words as though it were stored that way. It is not
   -- packing to words is what the decoder does on the way out. The pak is
   **1.89 MB** of 4-bit IMA ADPCM, 12.6% of the filesystem. Nothing is
   trimmed and nothing drops to 8 kHz.

4. **Volume/quality gate: one clip A/B'd at the desk before the rest are
   trusted.** Rendered and waiting:

       afplay corpus-voice/audition-pak/wav/ff7a366a.wav

   "Tell me more about your family." -- 2.51 s, median F0 **149.5 Hz** against
   the 148 Hz this document predicts for Ava at p3-warm, peak 15162, round-trip
   SNR 30.5 dB. Decoded *through the ADPCM round trip*, deliberately: an
   audition of the pre-encode render would be an audition of something the
   device never plays. The same clip and id are in the full pak.

   **Nothing here has been played through the board's speaker.** Per CLAUDE.md
   that gap is the one that shipped a bug before: `say` exiting 0 proves a file
   was written and nothing about what a person hears, exactly as an unpowered
   panel accepts SPI writes.

The container is `voice.pak` -- one file, one upload, seekable, bisected on
flash without ever being held in RAM. Its layout is normative in
`tools/voice_pak.py::write_pak` and mirrored in `src/adpcm.py`; the encoder is
checked against `audioop`, CPython's own IMA implementation, so that agreeing
with the decoder is evidence about the format rather than about two people
having made the same mistake.

Board state at pause: ELIZA deployed and working (TFLM backend, chatty 0.35
gate), six stopgap clips on flash, firmware `e7b1069a…` (16 MB + TFLM).
