# The synthetic corpus, and what its split can and cannot prove

> ## State as of 2026-08-18, end of session
>
> **The tree is coherent and `tools/corpus.py corpus-tts --takes takes,takes-oov
> --check` exits 0.** Anything measured against it is valid. It is *partial*,
> and what is missing is named below rather than left to be discovered.
>
> | | |
> | --- | --- |
> | Rendered | **16030 utterances, 14 voices** — 8 train, 3 val, 3 test |
> | Manifest | `corpus-tts/manifest.json`, indexes all 16030 |
> | Leak check | passes: no voice or family spans two splits |
> | `unknown` class | **complete** — 734 per voice, every voice |
> | Human held out | 22 (`takes/` 10 keywords, `takes-oov/` 12 negatives) |
>
> Rendered voices, all with the full 330 word / 734 unknown / 76 variant plan:
>
> | split | voices |
> | --- | --- |
> | train | Bruce, Samantha, Joelle (Enh), Noelle (Enh), Daniel, Tara, Tessa, Eddy (UK) |
> | val | Aman, Karen, Moira |
> | test | **Allison (Enh, en_US f)**, **Nathan (Enh, en_US m)**, Rishi |
>
> Test holds an American voice of each gender, which is what three roster
> rebuilds were for. American voices are in train too, which the first roster
> did not manage at all.
>
> ### What is missing, in the order worth doing it
>
> 1. **The roster is one generation behind the installed voices.** `Ava
>    (Premium)`, `Evan (Enhanced)`, `Susan (Enhanced)`, `Nicky` and
>    `Samantha (Enhanced)` installed after it was frozen and are **not in it**.
>    Evan matters most: it is a second American *male* natural voice, which
>    would allow one in train and one held out. Re-run
>    `python3 tools/say_voices.py freeze corpus-tts/roster.json`, then
>    `python3 tools/train_corpus.py corpus-tts/ --index` (which **re-files**
>    existing audio rather than re-rendering it), then render the new voices.
>    Note the rebuild will drop Compact `Samantha` in favour of
>    `Samantha (Enhanced)` and orphan its ~1140 files to `_stale/`.
> 2. **The 16-voice expressive family and 13 novelty voices are unrendered**
>    except Eddy (UK). That is ~28 voices × 1140 = ~32000 files, about 1.8
>    hours. They train only, so they cannot affect the split.
> 3. **`Aman` carries 70 files more than the plan** — 764 unknown and 116
>    variant against 734 and 76 — left over from an earlier plan generation at
>    rates the current rotation no longer picks. Harmless (valid audio, correct
>    labels, no leak) but it breaks the property that every voice contributes
>    identically. `--index` does not prune by `(text, rate)`, only by text.
>
> **A clean rebuild is a reasonable alternative to all three** and is about 2.5
> hours: `rm -rf corpus-tts` (mind the zsh glob trap below), re-freeze, render.
> Nothing in the tree is precious — it is entirely reproducible from
> `roster.json` plus the tools.


The training data for the speaker-independent recogniser. It is built from
macOS `say` voices, because the alternative is recording ten people for two to
three hours each and Google Speech Commands overlaps our vocabulary by exactly
two words — YES and NO, and NO was cut as a homophone of "know".

[speech-design.md](speech-design.md) argues why the recogniser is shaped the
way it is; [speech.md](speech.md) pins the feature contract;
[speaker-independent.md](speaker-independent.md) is the model that consumes
this. This file is about the **data**, and mostly about the ways it lies.

| | |
| --- | --- |
| Roster | `tools/say_voices.py` → `corpus-tts/roster.json` |
| Corpus | `tools/train_corpus.py` → `corpus-tts/` |
| Loaders | `tools/corpus.py` |
| Checks | `tools/test_corpus.py`, `python3 tools/corpus.py corpus-tts/ --check` |
| Noise | `tools/fetch_background_noise.sh` → `corpus_noise/` |

All of it is gitignored. Rebuild with:

```
./tools/fetch_background_noise.sh
python3 tools/say_voices.py freeze corpus-tts/roster.json
python3 tools/train_corpus.py corpus-tts/ --jobs 7
python3 tools/corpus.py corpus-tts/ --check
```

## 188 voices are not 188 speakers. They are 13.

This is the headline finding and it sets the ceiling on the whole approach.

`say -v '?'` lists **188 voices**, of which **48 are English**. Every one of
the 42 usable ones renders distinct audio — *measured*, 42 distinct SHA-256
digests over the probe utterance, no collisions. But distinct audio is not a
distinct speaker, and the gap is large:

| | voices | |
| --- | --- | --- |
| listed, English locales | 48 | |
| — singing rather than speaking | −6 | Bells, Cellos, Organ, Jester, Bad News, Good News |
| **usable** | **42** | in **19** prosody families |
| — `expressive`, one synthesiser | 16 | the iOS voices, 8 names × en_GB/en_US |
| — `novelty`, formant and effect | 13 | Fred, Zarvox, Bahh, Whisper, Boing … |
| **`natural`, modern concatenative** | **13** | Allison, Nathan, Joelle, Noelle (Enhanced en_US), Samantha, Bruce, Daniel, Karen, Moira, Tessa, Rishi, Aman, Tara |

Locales: 27 en_US, 9 en_GB, 3 en_IN, 1 each en_AU / en_IE / en_ZA. Qualities:
4 Enhanced, 38 Compact — **in the frozen roster**. `Ava (Premium)` has since
installed and is the first of that tier on this machine; see the state banner
at the top.

**Thirteen**, of which four are American Enhanced voices downloaded partway
through the session. That is how many genuinely human-sounding, mutually
independent English voices this Mac has, and it is the number the experiment is
really running on. Everything else is either one synthesiser wearing sixteen
timbres or a 1990s formant synthesiser.

It was **eight** before the download, all Compact, and only one of them en_US.
The roster is worth rebuilding whenever more voices land: `say -v '?'` grew
from 184 entries to 188 during this session as Enhanced voices finished
installing, and Ava, Evan, Nicky, Susan, Tom and Zoe were still downloading
when this was written. **Rebuild the roster before trusting the split**, and
note that doing so re-files the corpus rather than re-rendering it — see
`--index` below.

### How the tiers are decided, without listening to anything

Every rule is a measurement, because "sounds robotic" is not reproducible.

**Singing vs speaking** — median duration of five probe words at 175 wpm.
Measured: every voice that speaks lands in **407–911 ms**, every voice that
sings in **1185–1940 ms**. The cut at 1050 sits in empty space. It has to be
the *median*: the iOS voices take 1113 ms over `computer` alone.

**`natural`** — a prosody family of one *and* a median word ≤ 600 ms. On the
pre-download roster this selected exactly the eight modern voices, with a
160 ms gap either side. It is a **weaker rule now**: Bruce, a 1990s MacinTalk
voice, medians 564 ms and is admitted, and Enhanced arrivals sit at 497–564 ms
alongside it. Bruce is en_US and male and was the only such voice for a while,
so he earns his place in training regardless — but the tier boundary is no
longer the clean empty band it was, and a future rule should probably not rest
on duration alone.

**`expressive`** — a family whose members span more than one locale. Exactly
one family does.

### The three ways the voice list lies

**1. Voices share a synthesiser, and nothing says so.** Per-word durations at
175 wpm:

```
Eddy / Flo / Grandma / Grandpa / Reed / Rocko / Sandy / Shelley, en_GB:
    mother 830  father 911  computer 1113  yes 814  sad 975
the same eight names, en_US:
           878         943            1065     814      975
```

Eight voices, one duration to the millisecond. One prosody model with eight
timbres. A model that learns *timing* rather than *timbre* scores perfectly
across all eight and has learned nothing that transfers to a person. The legacy
voices do it in smaller groups: Fred/Trinoids/Zarvox/Bubbles share a tuple,
Junior/Kathy/Ralph/Superstar another, Albert/Bahh a third.

**2. Two are the same voice under two names.** Long-term average log-mel
distance, Q8 log2, over 903 pairs — the median pair is **220.5**, the widest
**643.7**:

| | |
| --- | --- |
| Eddy (UK) ↔ Reed (UK) | **6.9** |
| Eddy (US) ↔ Reed (US) | **10.0** |
| Grandma (US) ↔ Shelley (US) | 39.5 |
| Sandy (UK) ↔ Sandy (US) | 61.5 |
| Eddy (UK) ↔ Eddy (US) | 63.1 |

Note the last two: **the accent setting matters more than the name**. Reed is
Eddy. Flo (UK) and Flo (US) are one vocal tract.

**3. `say -v <name>` does not fail on a voice that is not installed — it
renders in the system default voice and returns 0.** A roster built from a
written-down list of names silently becomes one voice under many names, those
names land on both sides of the split, and the score becomes a voice matched
against itself. **It reads as a good result.** Found by `si-model` in a
throwaway corpus where four voice names shared one MD5.

This roster is built by probing `say -v '?'`, which lists only what is
installed, so it is structurally immune — and `say_voices.assert_distinct`
checks it anyway and **raises**, at `freeze` time and again at the top of every
`train_corpus.py` build, so it is not a step anybody can forget.

**And the list is not stable.** Two `say -v '?'` runs eleven minutes apart
returned 184 then 186 lines, the second containing `Aman (English (India))` and
`Tara (English (India))` **twice each** — Siri voices had finished downloading
in the background. `say -v` takes a name, so the second of a colliding pair is
**unaddressable**. That is why `roster.json` is frozen to disk and the builder
reads *that*, never the live list.

## The split is chosen against requirements, not by share

**Whole voices are held out, never individual utterances** — and the unit is
the *family*, not the voice name.

The natural tier is assigned against explicit requirements, because a 3:1:1
greedy pass satisfies them only by luck, and the first roster that used one
produced a split with **no American English voice in training at all**: every
natural training voice was en_IN, en_IE or en_ZA, and the only natural en_US
voice sat in `val`. The user speaks American English. So the model and the DTW
control had never heard the accent they were tested on, and **an accent gap
could have been measured and reported as a speaker gap.**

The requirements, in the order they are satisfied:

1. **An en_US voice of each gender held out for test** — best quality first,
   so generalisation to an unseen American speaker is measured rather than
   assumed. Test is where two or three voices carry the whole claim; train has
   thirty others to dilute a weak one.
2. **An en_US voice of each gender in train** — train on the accent that will
   be spoken to it.
3. **Both genders in train**, filled from other locales if en_US cannot: the
   user is male and wants the toy to work for women too, and an accented voice
   of the right gender beats no voice of that gender.
4. **Both genders in val and test.**
5. Whatever remains, 3:1:1.

Gender is **curated** in `VOICE_GENDER`, not measured — `say -v '?'` reports a
locale and nothing else. Anything absent is `unknown` and satisfies no
constraint, because a requirement met by a guess is worse than one reported
unmet. The iOS expressive voices are deliberately absent from the table: Apple
does not present Eddy, Flo, Reed, Rocko, Sandy or Shelley as gendered.

Quality comes from the name — `Allison (Enhanced)`. Enhanced and Premium are
much better models than the Compact default, so they are spent on the held-out
American slots first. `name_stem` collapses `Allison (Enhanced)` onto
`Allison`, so a voice installed at two qualities is one speaker for splitting.

### Requirements that cannot be met are reported, not worked around

`check_requirements()` returns complaints and `roster.json` carries them in
`unmet_requirements`. Some are unsatisfiable by any assignment: with exactly
one en_US male voice installed, it can be trained on or held out, never both.
Picking one silently and calling the split satisfied would be the worse
failure. Currently unmet: **no en_US voice in val** — all four natural en_US
voices are spent on train and test, which is the right priority.

`group_families` takes connected components over three relations: same prosody
(every probe word within 8 ms), same name stem (`Flo (UK)`/`Flo (US)`), and
twin timbre (fingerprint < 25). Any one of them means one speaker.

| split | voices | natural voices |
| --- | --- | --- |
| train | 36 | Bruce (en_US m), Samantha (en_US f), Joelle (en_US f, Enh), Noelle (en_US f, Enh), Daniel (en_GB m), Tara (en_IN f), Tessa (en_ZA f) |
| val | 3 | Aman (en_IN m), Karen (en_AU f), Moira (en_IE f) |
| test | 3 | **Allison (en_US f, Enh)**, **Nathan (en_US m, Enh)**, Rishi (en_IN m) |

Only `natural` voices are held out; `expressive` and `novelty` always train. The
held-out voices are the entire synthetic evidence for generalisation, and a
test split containing Bahh predicts how the model does on a sheep noise.

Test holds an American voice of **each** gender, both Enhanced, which is the
arrangement the whole roster rebuild was for.

### The first split was dishonest, and the failure is worth keeping

The first roster grouped by prosody alone. That put the eight en_GB expressive
voices in `train` and six of the eight en_US ones in `val` — so `val` measured
generalisation to Flo having trained on Flo. **17 of 34 close pairs straddled
the boundary.** Nothing errored; the val number would simply have read several
points high, and the number is what the experiment reports.

`check_straddle` now **raises** if any pair below 75 lands on opposite sides.
It is a backstop rather than the defence — name-stem grouping makes those pairs
unable to straddle by construction — and it exists to catch a future change to
the grouping.

The threshold sits between two populations: the dishonest pairs run 61.5–73.3,
and the closest genuinely-different pair that survives grouping is **Aman ↔
Tara at 84.7**. Those two *are* allowed to straddle, and it is better that they
do: Aman trains and Tara tests, so the en_IN accent appears on both sides. A
test accent absent from training would measure accent transfer, which is a
different and harder question than the speaker transfer being measured.

### Known weaknesses, stated rather than buried

- **val and test are two voices each.** That is a wide error bar, and it is not
  fixable by re-splitting: there are only eight natural voices in total.
  Reporting a difference between two architectures from n=2 held-out voices is
  not something this corpus can support.
- **train is 16/33 one synthesiser.** The expressive family is 43% of the
  usable voices and they share a duration model exactly.
- **Every voice is a consistent talker.** Synthetic speech varies in *session*
  — rate, level, channel — and never in *articulation*: no mumbled vowel, no
  swallowed final consonant, no day with a cold. This is the same caveat
  `docs/speech.md` puts at the top of its own accuracy figures, and it applies
  here more strongly, not less.
- **`takes/` is a handful of utterances and every one of them clips.** A
  held-out synthetic voice proves the model did not memorise a synthesiser. It
  does **not** prove the model recognises a person, and the two are different
  claims — so this set is the only one that settles it. See below.

### The endpointer needs a 5.71:1 signal-to-background ratio, and gain cannot buy it

The single most useful thing measured this session, and it was nearly
misdiagnosed. `src/vad.py` finds a word only if some frame reaches `ITU`. From
`thresholds()`:

```
ITL = min(3*(IMX-IMN)/100 + IMN, 4*IMN)      ITU = 5*ITL
```

The `4*IMN` branch only binds above `IMX/IMN = 101`, so in practice
`IMX >= 5*(0.03*(IMX-IMN) + IMN)`, which solves to

> **`IMX/IMN >= 4.85/0.85 = 5.706`** — the loudest frame must be 5.71x the
> background, or the utterance is discarded before any recogniser sees it.

**That ratio is gain-invariant.** `IMX` and `IMN` are both sums of `|x|`, so
scaling every sample multiplies both and cancels. Verified by halving all
samples and recomputing: 32.12 before, 32.12 after, to three figures.

**So turning the input gain down does nothing at all for this**, which is the
opposite of the advice an earlier draft of this file gave. The ratio moves only
if the speech gets louder *relative to the room* — closer to the mouth, or a
quieter room — or if `ITU`'s multiplier moves. (Derived by the model-side agent
from the algebra; independently confirmed here against the recordings.)

The predicted cutoff separated a first batch of takes perfectly: rejected at
ratios 3.6/3.7/4.4/4.9/5.5/5.7, found at 6.3/7.1/7.9/8.6. Nothing was near
5.706 by accident.

It also lines up with the corpus measurement two sections down — 49% VAD
failure at 8 dB SNR, 83% at 6 dB, 0% at 11 dB and above. Two independent
routes, synthetic additive noise and real room noise, both land on the ratio.
Every VAD threshold in [speech.md](speech.md) was tuned on clean single-voice
synthetic audio at high SNR, and that is the thing to suspect.

### The takes themselves are fine now, and the cause was neither thing we said

Re-recorded at 18:16 on 2026-08-18, and the set was still growing while this
was written — treat the counts as a snapshot, not a total. Measured over the
eight present at the time:

| | |
| --- | --- |
| `src/vad.py` finds the word | **8 of 8** |
| `IMX/IMN` | 22.6 … 49.3, against a floor of 5.71 |
| endpointed length | **320–460 ms** |
| the same words, synthetically | 300–640 ms |

The lengths now sit inside the synthetic range, which is the check that
matters: an endpointer returning a plausible, word-dependent duration is doing
its job, and one returning the same length regardless of word is the failure
[speech.md](speech.md) warns about.

**The cause was the activation chirp, not the room and not the gain.** The tone
rings ~180 ms and decays over a further ~140 ms, and `src/vad.py` estimates its
background from the **first 100 ms** — so the tail of the chirp was being
measured as the noise floor. That inflates `IMN`, which is the denominator of
the 5.706 ratio, and no amount of speaking or gain-setting could beat a floor
made of the device's own beep. `src/listen.py` now waits it out
(`CHIRP_SETTLE_MS = 140`); its comment records 14 of 22 recordings rejected
before the fix.

Worth keeping because of how it was found. Two of us measured the same failure
from different sides — synthetic SNR sweeps here, real captures there — derived
a correct threshold condition from the algebra, and then each attached a
plausible wrong cause to it: "the gain is too high" and "the room is too loud".
Both were confidently argued, one was gain-invariant and provably could not
have been it, and the actual fault was a bug one stage upstream that neither
measurement could see. **The derivation was vindicated and both conclusions
drawn from it were wrong.** The `docs/speech-design.md` habit of separating
*measured* from *estimate* would not have caught this either: every number
involved was genuinely measured. What was inferred was the causal story
attached to them.

**The clipping has not gone away and is a genuine gap.** Every take peaks at
full scale, 12 to 778 saturated samples. It is not what was defeating the
endpointer — `dream` clips only 12 samples and passes, and the heavily clipped
takes pass too — but the training corpus draws gain and tilt and **never
saturation**, so real audio carries a distortion no synthetic example contains.
That is an unmodelled augmentation axis, independent of everything above, and
the cheapest thing to add next.

## Indexing audio that already exists

```
python3 tools/train_corpus.py corpus-tts/ --index
```

`manifest.json` is the authority and nothing downstream reads the directory
layout, so an interrupted build leaves a tree full of usable audio that is
completely unusable. `--index` walks what is on disk and writes the manifest
without re-rendering anything — two hours saved when the WAVs are sound.

**It refuses to bless a stale split.** The split lives in the path, so a tree
written before a roster change disagrees with the roster now, and indexing it
blindly would record a leak as fact. Any file whose directory disagrees with
`roster.json`, or whose text is no longer in the plan, is moved to `_stale/`
and reported.

That is not hypothetical: it fired the first time it ran. The tree held two
builds layered on top of each other, and **Daniel was physically present in
both `val/` and `test/`** — 1142 files in `test/` from the pre-tier roster
while the current one puts Daniel in `val/`. Nothing else would have noticed;
`corpus.py --check` reads the manifest, so a manifest written over that tree
would have passed the check *and* leaked.

The reason the old tree survived at all is worth recording. A `rm -rf` meant to
clear it was written as `rm -rf a b c corpus-tts/_raw.*.wav`, the glob matched
nothing, and **zsh aborts the whole command on a failed glob** rather than
passing it through as bash would. The output was one line — `no matches found`
— which reads as a warning and was not. Nothing was deleted.

`prefer_quality()` dropping the Compact twin is worth noting as a case where
one invariant paid off twice: grouping by **name stem** was written for locale
variants (`Flo (UK)` / `Flo (US)`) and turned out to handle quality tiers
(`Samantha` / `Samantha (Enhanced)`) for free, before that case existed.
`assert_one_split_per_stem()` restates the guarantee as an assertion anyway —
not redundant, because the guarantee is one edit to `name_stem` away from
disappearing silently.

What `--index` cannot recover is the per-utterance augmentation: gain, tilt,
SNR and noise source are choices, not properties of the samples, so they are
recorded as **null** rather than guessed. `samples` and `endpoint_ms` are
recomputed from the audio and are real. Records carry `indexed: true` so a
consumer can tell the two apart. A later full or resumed build merges over
them, replacing nulls with the real values for anything it re-renders.

## What is in it

42 voices × 1140 utterances = **47880 files, about 2.0 GB**. Budget roughly
2.5 hours at `--jobs 7`: `say` scales poorly, giving about 3x for 7 workers,
and the build runs at about 5 files/s. The natural tier alone is 13 voices and
about 50 minutes, and it covers every val and test voice, so it is the subset
worth building first. Layout is `corpus-tts/<split>/<category>/<voice>/<text>.r<rate>.<n>.wav`,
and `manifest.json` is the authority — one record per utterance carrying path,
split, category, label, text, voice, family, tier, locale, rate, and every
augmentation parameter.

Three categories, because lumping them together hides the collision that
matters:

| category | | scored as |
| --- | --- | --- |
| `word` | 22 spoken forms in 21 classes | hit or miss |
| `unknown` | 298 texts that must stay silent | a fire is a **false positive** |
| `variant` | inflections of a live keyword | neither — see below |

A `variant` is MOTHER'S, WORKED, LOVES. Firing there is DOCTOR working
correctly, so counting it as a false positive would drag the rejection
threshold down to punish good behaviour. The engine matches exact forms, so
LOVE firing on "loves" is arguably right.

### Sweeping a threshold without negatives inflates recall

A precision/recall sweep run over the keywords alone is not a weaker
measurement, it is a wrong one. **With no negatives present nothing can fire on
an ordinary word**, so precision stays at 1.000 however low the threshold goes,
the sweep keeps lowering it, and it reports the recall that buys. Restore the
negatives and the same threshold fires on one of them.

Measured on the human set: evaluating `takes/` alone reported recall 0.700 at
precision 1.000; adding the twelve negatives in `takes-oov/` put the honest
operating point at **recall 0.500, precision 1.000**, with 0.700 costing
precision 0.875. Same model, same audio, no code changed — only whether the
things it must stay silent on were in the room.

**The tell is that the flattering number arrives with a *lower* threshold.**
That is worth memorising as a general rule: a rejection threshold that drifts
down while the score goes up means the thing that was holding it up has left
the evaluation set. It is the same shape as the `say_corpus.RETIRED` trap two
sections up — a test set that silently stops containing the case it exists to
test — and both are invisible in every curve they affect.

So: score against `word` **and** `unknown` together, always. `variant` is
scored as neither.

`NEAR_MISS` groups 95 of the unknowns by the keyword each attacks — *other*,
*another* and *wonder* against FATHER; *smother* and *mutter* against MOTHER;
*know* against the retired NO. They get twice the rendering budget of an
ordinary noun, because they are where a false fire comes from and false fires
are what the design trades recall away to avoid. Retired words (`no`, `want`,
`need`) get the full rate set, for the reason `say_corpus.RETIRED` gives.

### Augmentation

Waveform domain only. Feature-domain augmentation (SpecAugment, frame shift)
belongs to the model side, and doing it twice would be worse than doing it
once.

| axis | range |
| --- | --- |
| speaking rate | 145, 160, 175, 190, 205 wpm |
| gain | −9 … +3 dB |
| spectral tilt | −0.3 … +0.3, first order |
| additive noise | SNR 8–30 dB against the utterance |
| lead silence | 200–600 ms, drawn per utterance |

**Noise is specified as SNR, not as an absolute level**, because the gain draw
moves the utterance and a fixed floor would make the achieved SNR depend on it
— loud renditions clean, quiet ones buried, with the label correlated to level
rather than to the word. The achieved absolute dBFS is recorded anyway.

The board's own captures measure **mean|x| ≈ 1200 of 32767, about −28 dBFS**,
which is a high floor; with speech at mean|x| 3000–6000 its working SNR is
8–14 dB. The augmentation is centred there rather than on the −46…−34 dBFS
`say_corpus.py` uses for its quiet close-talk evaluation channel.

Real recorded noise is used when `corpus_noise/_background_noise_` is present —
six recordings from Speech Commands, CC BY 4.0 — because broadband hiss is the
easiest interference a spectral front end ever meets. It falls back to the LCG
noise and records which was used.

**The lead silence is randomised** so a convolutional model cannot learn where
in the frame the answer starts. That is free accuracy here and none on the
board, where the VAD decides.

### Two measurements that changed the design

**The endpointer gives up below 10 dB.** Over 330 in-vocabulary utterances:

| SNR | 30 | 24 | 18 | 14 | 11 | 8 | 6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `src/vad.py` cannot find the word | 0% | 0% | 0% | 4% | 0% | **49%** | **83%** |

So the 6 dB tier was dropped — it is mostly files the DTW path can never use.
8 dB is kept deliberately: it is inside the board's own range, and an utterance
the VAD rejects is not wasted, because on the device that rejection *is* the
designed behaviour. Every record carries `endpoint_ms`, or null, so a template
matcher can filter and a fixed-window classifier need not.

**The LCG collapsed the noise selection to half the files.** `sc.Rng` is an LCG
modulo 2³¹ and the noise file is chosen at the same offset in the stream every
time. Measured over 2000 utterances with the build's actual draw pattern:

```
rng.pick     {0: 616, 2: 708, 4: 676}      three files, never six
top 15 bits  {0: 331, 1: 339, ... 5: 317}  uniform
```

The first corpus used only `running_tap`, `exercise_bike` and
`doing_the_dishes`. Nothing reported it — the manifest faithfully recorded the
noise for each file, and it took reading the histogram to notice half the
column was missing. `pick_hi` takes the top bits.

## The rule about `takes/`

`takes/` is real human speech through the board's own microphone, written by
`tools/enrol.py`. It is loaded under the split name `human`, which
`split_for_training()` refuses to return and `load()` refuses to read —
`load(root, "human")` raises rather than returning an empty list. A caller has
to name `load_takes()` deliberately. There is no flag that mixes them.

That is more friction than a boolean, and it is the right amount, because the
failure it prevents is silent: a model trained on its own test set reports a
number that looks like success.

`load_takes()` returns `(records, note)` and **degrades** when the directory is
not there — an empty list and a stated reason, never an exception and never a
silent zero that reads as a passing test. `--check` exits non-zero when there
are no human takes, so "speaker independence is untested" is a visible state
rather than an absent one. Labels are correct by construction (`enrol.py` names
the file after the word it asked for, and nothing is transcribed); SICK maps to
the SAD class exactly as it does on the device, and a word recorded with
`--allow-any` becomes an `unknown` record rather than being dropped.
