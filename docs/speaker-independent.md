# A speaker-independent spotter — design notes and where it got to

Can the ball recognise **anybody**, without a human recording a keyword
campaign? [speech-design.md](speech-design.md) costed a trained CNN against DTW
and chose DTW on one row of the table — **recordings per word**, 50-100 against
DTW's 3. The idea under test here is that macOS `say` makes that row free: train
on synthetic voices, and the spotter stops being speaker-dependent.

This document records what was built, what was measured, and what is still
unknown. It worked, on the one speaker it has been tried on.

> ## The result, in three numbers
>
> CNN trained on the 8-voice natural tier (4 training voices, **no American
> voice**), int8, evaluated on 22 real utterances — 10 keywords and 12
> negatives — from one American speaker through the board's own microphone. DTW
> control re-run on the later American-inclusive roster. Every figure measured,
> none estimated.
>
> | | top-1 | precision 1.000 at recall |
> | --- | --- | --- |
> | CNN, synthetic held-out voices | 0.626 | 0.013 |
> | **CNN, real speaker** | **0.700** | **0.500** |
> | DTW control, same real speaker, best of four configurations | 0.700 | **0.000** |
>
> **Identical top-1 — and only one of them is usable.** At a threshold where the
> CNN never once fires wrongly it catches half the keywords, while DTW with
> synthetic templates has no such threshold at all: at every setting tried, its
> first correct fire arrives no earlier than its first false one. The difference
> between the two is not recognition, it is **rejection**.
>
> The model also recognises the real human at least as well as it recognises
> held-out synthetic voices, which the design had predicted would not happen.
>
> **n=10 keywords and 12 negatives, one speaker, one session, one room.** Half
> of ten is five utterances; the error bar is enormous. This answers "does a
> synthetically-trained model recognise a real person at all" — which was the
> question — and nothing finer.
>
> **The CNN rows were measured on the `.tflite`, not on what the board runs.**
> The device executes a `.tmdl` through TinyMaix, an independent
> reimplementation whose arithmetic differs: top-1 disagreed on **3 of 8**

> **A caveat on "host TFLite" as a reference, measured later** (see
> [tflm-usermod.md](tflm-usermod.md)): `tf.lite.Interpreter` is three runtimes
> that disagree among themselves -- XNNPACK (the default), optimised CPU, and
> the reference kernels differ by up to 8 of 256 output counts on these same
> cases, with only the reference kernels bit-identical to TFLM. Every top-1 in
> those comparisons agrees across all three, so the TinyMaix finding stands,
> but the *size* of its divergence was measured against a reference with up to
> 8 counts of movement of its own. Comparisons that need exactness must name
> the kernel set, not just "TFLite".
> patches. DTW is untouched by this, so the CNN-against-DTW rows are not
> like-for-like and the CNN's 0.500 is provisional against them. What survives
> is that DTW has **no** usable threshold in any configuration, which nothing on
> the CNN's side can change. See
> [Tuning against the wrong model](#tuning-against-the-wrong-model).
>

## Start here — what to run, in order

Everything below this section is the reasoning. This is the doing. The corpora
and `build/` are gitignored, so a fresh machine rebuilds them from these
commands.

```
# 0. environments (two, deliberately -- TinyMaix's converter needs TF < 2.14)
uv venv --python 3.11 .venv     && uv pip install --python .venv/bin/python "tensorflow>=2.16,<2.21"
uv venv --python 3.10 .venv-tm  && uv pip install --python .venv-tm/bin/python "tensorflow-macos==2.13.0" "numpy<2" pillow

# 1. corpus -- roster is committed, audio is not
python3 tools/train_corpus.py corpus-tts/ --jobs 7        # ~2 h full, resumable
python3 tools/train_corpus.py corpus-tts/ --index         # rebuild the manifest
python3 tools/corpus.py corpus-tts --takes takes,takes-oov --check

# 2. the baseline, first -- needs no training
.venv/bin/python tools/si_dtw_control.py corpus-tts/manifest.json \
    --voices 5 --takes takes --takes-oov takes-oov

# 3. train, then evaluate against BOTH takes directories
.venv/bin/python tools/si_train.py corpus-tts/manifest.json --cache <dir> --epochs 80 --out build/si
.venv/bin/python tools/si_eval.py build/si.tflite corpus-tts/manifest.json --takes <keywords+negatives>

# 4. device artefact
cd TinyMaix && ../.venv-tm/bin/python -m tools.tflite2tmdl \
    ../build/si.tflite ../build/si.tmdl int8 1 80,26,1 22
python3 tools/tmdl_info.py build/si.tmdl --pad
```

**`--takes` must point at keywords *and* negatives together.** A keywords-only
directory cannot observe a fire on an ordinary word and reports a flatteringly
low threshold; see [the real-speaker result](#the-real-speaker-result).

### What is on disk, and what each thing is

| path | what | gitignored |
| --- | --- | --- |
| `build/si_real.tflite` `.keras` `.json` | the trained int8 model; the `.json` carries class order, frame count, band count, input shift and quantisation params | yes |
| `corpus-tts/roster.json` | the frozen voice roster, splits, tiers, fingerprints | **no** |
| `corpus-tts/` audio + `manifest.json` | the synthetic corpus | yes |
| `takes/` `takes-oov/` | **the test set** — 10 keywords, 12 negatives, `MIC_GAIN = 1`, no clipping, all endpoint | yes |
| `corpus-tts/_stale/` | quarantined files from a superseded roster. **Never read these.** | yes |

Two further capture conditions existed — the same words at `MIC_GAIN = 3`
(every take saturating) and the original captures with a chirp tail in the
first 300 ms. They were what made the chirp and the clipping answerable as
controlled comparisons rather than as inferences; both questions closed
(12a942e, 61967bc), and the recordings were then **destroyed at the user's
request, including from git history** — they were voice recordings whose
purpose was spent. The comparison *numbers* survive in those commits and in
this document. `takes/` and `takes-oov/` are kept locally, gitignored, for
the CNN evaluations.

### The next thing to do, and it is not architectural

**Speakers nobody trained on.** Every conclusion here rests on ten keywords and
twelve negatives from **one** person, who is also the person the thresholds were
tuned against. For a workshop that is the wrong shape of evidence entirely: the
question is not "does it recognise its owner" but "does it stay quiet around
strangers".

No change to the model is worth as much as five more voices, and collecting them
needs nobody from this team.

#### The protocol, for the user to run without us

**Three minutes per person.** Ask a colleague, a housemate, anyone who has not
been recorded yet. Different rooms and different microphone distances are a feature,
not a problem to control away.

```
# keywords -- one take of each, in the vocabulary's own words
uvx --from pyserial python tools/enrol.py speakers/<name>/ --reps 1

# negatives -- the words that must NOT fire, in the same session
uvx --from pyserial python tools/enrol.py speakers/<name>-oov/ --allow-any --reps 1 \
    --words other,another,wonder,mothers,brothers,know,want,need,today,problem,friend,because
```

(`--allow-any` is what permits words outside `vocab.FORMS`; without it the tool
refuses them, which is deliberate — it once stopped someone recording templates
for a retired keyword.)

Then, for each speaker:

```
.venv/bin/python tools/si_eval.py build/si_real.tflite corpus-tts/manifest.json \
    --takes <a directory holding BOTH that speaker's sets>
```

Four things make the data usable rather than merely collected:

1. **Keywords and negatives from the same person in the same session.** A
   precision figure needs both; see [the real-speaker
   result](#the-real-speaker-result) for what happens when only keywords are
   scored.
2. **One take each is enough at first.** Five speakers x 22 utterances beats one
   speaker x 110, because the variance that matters is between people.
3. **Check `peak` and `clipped` in the printed stats.** `MIC_GAIN = 1` should
   give peaks well under 32768 and zero clipped samples. If it clips, the
   recordings still endpoint but they carry a distortion the corpus does not
   model.
4. **Do not tune anything on a speaker until every speaker is collected.**
   Thresholds tuned on the people you have are thresholds fitted to them.

**What five speakers would settle:** whether precision 1.000 survives contact
with voices the model has never seen and nobody tuned against — which is the
entire question a workshop build asks, and the only one this document cannot
currently answer.

## The finding: rejection, not recognition

DTW and the CNN reach **identical top-1 on the real speaker — 0.700 each** — and
only one of them can be deployed, because only one can decline to answer. At
every threshold tried, in four template configurations, **DTW's first correct
fire arrives no earlier than its first false one**; the CNN has a setting where
it catches half the keywords and fires wrongly on nothing at all, keywords and
negatives alike.

The firm half of that is DTW's. Its figures are measured on the arithmetic it
actually runs, and nothing about the CNN can hand DTW an operating point it does
not have. **The CNN's half was measured on the `.tflite`, not on the `.tmdl` the
board executes** — so *how much* better it is remains open, and the achievable
curve under TinyMaix may differ in shape and not only in placement. See
[Tuning against the wrong model](#tuning-against-the-wrong-model).

That is the whole result, and it is not the one the experiment set out to find.
The question was whether synthetic voices transfer to a human. They do, for both
approaches, about equally. What separates the two is that a trained `unknown`
class can represent "none of these" and a nearest-neighbour distance cannot —
`argmin` always returns a word.

`docs/speech.md` argued exactly this from first principles, before any of it was
measured: *"Rejection is a feature, not a safety net"*, and the honest metric is
precision and recall separately because a single accuracy figure averages
together the one error that matters and the one that does not. This is that
argument turning into a number. Two systems, the same accuracy, opposite
verdicts.

Note what that costs the usual way of reporting this: on a single accuracy
figure these two systems are **indistinguishable**, and the whole difference
lives in the axis that figure averages away.

## The first fully-labelled live session

Every earlier number came from stored recordings. This is the device, a real
speaker, one word per press, ground truth from the person speaking. All captures
clean, 0 clipped across 24 turns. **n=1 take per word, one speaker, one session**
— but every row has a label, which none of the earlier live data had.

**Keywords, first press (19):**

| | |
| --- | --- |
| fired correctly (10) | mother 0.461, sister 0.836, wife 0.902, children 0.891, death 0.898, love 0.656, happy 0.883, yes 0.938, always 0.969, sorry 0.492 |
| correct but **tie-gated** (3) | father 0.500, husband 0.492, dream 0.492 — all margin 0.000 |
| missed to unknown (4) | brother 0.871, sleep 0.539, sad 0.957, work 0.430 |
| **false fires (2)** | **money → wife 0.758**, computer → children 0.441 |

**Negatives (5):** four clean rejections — "I wonder" 0.980, "I don't know"
0.867, "problem" 0.961, "hello there" 0.957 — and **"another" → brother 0.473**,
the near-rime attack landing on the CNN exactly where `docs/speech.md` recorded
it landing on DTW.

**With retries pooled: 13 of 19 words fired correctly at least once.** Brother
and computer never did; money never did, because it steals to wife.

### The misses are take-sensitive, not word-specific

The first reading of this table was that the misses were stable per word. **They
are not.** On a single retry, sad, sleep and work all fired — sad 0.809 having
missed at 0.957, work 0.809 having missed twice, sleep 0.367 scraping over. The
same speaker, the same word, a different press.

That matters because it changes the fix. Per-word augmentation is not indicated;
**delivery variance is**, which is the same conclusion the endpoint-sensitivity
measurement reached from the other direction — 50 ms of endpoint movement flips
a verdict. Two independent routes to one answer: the model is brittle to *which
audio the endpointer hands it*, not to particular words.

It also means **any per-class recall from single takes is close to meaningless**
and should carry an explicit "one take, plus or minus everything" caveat. The
multi-take evaluation is now more valuable than another architecture.

### brother is the clearest retrain target in the data

Stated carefully, because the pairing is the point:

- **brother: 0 for 3.** The only genuinely stable miss — 0.605, 0.871, 0.742,
  all to unknown.
- **"another" → brother: 2 of 3**, at 0.473 and 0.406, both marginal, both
  clearing the gate; rejected once at 0.828.

**The model never recognises this speaker's "brother" and fires `brother` on his
"another" two times in three.** The class as trained matches the wrong word.

The synthetic data predicted it: "another" fires brother in **4 of 11**
utterances, and `brother` receives wrong predictions from `mother` 19 times and
`father` twice — the family cluster `docs/speech.md` identified from phonetics
before any of this existed.

Unlike money→wife, these false fires are **weak** (p <= 0.473, margin <= 0.051),
so a modest margin floor may fence them as an interim measure where it could not
fence money.

### The tie gate: forgiving a tie is not free, but a probability floor makes it nearly so

Three correct answers were deleted by the margin gate on exact ties, which
suggests forgiving any tie whose argmax is not `unknown`. Measured against 4106
must-stay-silent and 1858 in-vocabulary synthetic utterances, that is a bad
trade — and adding one condition turns it into a good one:

| a tie passes when top-1 p >= | new correct | new false fires | new wrong-keyword |
| --- | --- | --- | --- |
| 0.00 (forgive any tie) | +19 | **+79** | +9 |
| 0.45 | +10 | +31 | +2 |
| **0.49** | **+3** | **+1** | **0** |
| 0.55 | 0 | 0 | 0 |

Open, it is **4.6 wrong fires for every right one**. It looked free live because
nineteen utterances cannot see a seventy-nine-case failure mode.

**With the floor it recovers all three live tie-gated answers** — father 0.500,
husband 0.492, dream 0.492 — while excluding both marginal false fires
(computer→children 0.441, another→brother 0.473).

Why a floor works at all: at 1/256 output resolution a tie at p ~ 0.5 means two
classes took ~128 counts each and everything else ~0 — a confident two-way split.
A tie at 0.28 means the mass is spread and nothing stands out. But the
distributions **overlap** — tying negatives median 0.441, tying correct keywords
median 0.453 — so 0.49 is a tuned operating point, not a clean separation, and
belongs beside `THRESHOLD` and `MARGIN` where it can be re-tuned.

### Every live confusion was already in the corpus

This is the methodological finding of the session, and it is a failure of
reporting rather than of data:

| live confusion | rate in synthetic val+test |
| --- | --- |
| money → wife | **16.0%** (13 of 81) |
| "another" → brother | **36%** (4 of 11) |
| computer → children | 7% (6 of 83) — did not reproduce, treat as one-off |
| "other" → father | **50%** (5 of 10), not yet seen live |

**Three for three on the confusions that reproduced.** The corpus knew about all
of them. The evaluation printed per-class *recall* at the recommended threshold
and never a confusion matrix — so a one-in-six directional collision sat in data
I had, unnamed, until a user heard it.

**The general form, and it is worth more than the instance: a per-class recall
table shows what each class misses and hides what each class steals.** Recall is
computed down the rows of a confusion matrix; false fires live in the columns.

## The retrain: money folded into unknown, endpoint jitter added — and it did not help

Both fixes implemented and measured. `money` folded into `unknown` (210
utterances, kept in the corpus so the model learns to *reject* the word rather
than never seeing it), plus **endpoint-jitter augmentation**: each training
utterance re-extracted at two additional spans with each edge perturbed
independently by up to 8 frames, 17000 extra patches.

| | before (`si_real`) | after (`si_jit`) |
| --- | --- | --- |
| synthetic val top-1 | 0.850 | **0.873** |
| **real speaker top-1** | **0.700** | **0.600** |
| real precision-1.000 recall | 0.500 | 0.500 |

**Better on synthetic, worse on the real speaker.** Exactly the pattern the width
sweep produced, and the second demonstration that synthetic accuracy does not
select a model here.

What it fixed and what it broke, per file:

| fixed | broke |
| --- | --- |
| "another" → unknown 0.977 *(was brother)* | **"wonder" → mother 0.793** — a new, strong false fire on a negative |
| "other" → unknown 0.887 | wife → **work** 0.680 *(was correct)* |
| "brothers", "mothers" → unknown | mother → unknown 0.809 *(was correct at 0.656)* |
| money removed as a fire source entirely | dream → unknown 0.832 |

So the near-rime family — another, other, brothers, mothers — is genuinely
fixed, and `money → wife` is gone by construction. In exchange the model
acquired `wonder → mother`, which is the *same near-rime attack it just learned
to reject*, arriving from a different direction and at higher confidence than
any of the ones it fixed.

**This model is not shipped.** `si_real` remains the deployed one.

### What the retrain actually established

Not that jitter augmentation fails — that is not measurable here. **That the
real test set cannot tell.** Real top-1 0.700 against 0.600 is one utterance out
of ten; the false-fire changes are one or two out of twelve. Every number moved
by less than the noise floor of a ten-word test set.

This is now the third independent route to the same conclusion — the width
sweep, the per-word miss pattern that turned out to be take-sensitivity, and now
a substantive retrain. **The binding constraint on this project is not the
model, the corpus, or the device. It is that there are ten keywords and twelve
negatives from one speaker, and that is too few to detect any change worth
making.**

Everything in [the protocol](#the-protocol-for-the-user-to-run-without-us) is
downstream of that. Five speakers would not merely add confidence to these
numbers; they would be the first evidence capable of ranking two models at all.

## The sweep, and what it says for a room of strangers

`--unknown-weight` and `--width`, on the full 16030-utterance corpus (8 training
voices, 4 of them American), evaluated three ways: held-out synthetic voices
(**val**, accented only), unseen **American** synthetic voices (**test**), and
the real speaker. Every figure from the int8 model.

| config | val top-1 | test(US) top-1 | real top-1 | val P=1.00 R | test P=1.00 R | **real P=1.00 R** |
| --- | --- | --- | --- | --- | --- | --- |
| w1.0 unk0.5 | 0.880 | 0.856 | 0.700 | 0.000 | 0.000 | 0.200 |
| **w1.0 unk1.0** *(shipped)* | 0.850 | 0.821 | **0.700** | 0.000 | 0.000 | **0.500** |
| w1.0 unk2.0 | 0.779 | 0.757 | 0.500 | 0.051 | 0.000 | 0.500 |
| w1.0 unk4.0 | 0.669 | 0.586 | 0.400 | 0.098 | 0.044 | 0.400 |
| w1.5 unk0.5 | **0.890** | **0.864** | 0.600 | 0.000 | 0.000 | 0.400 |
| w1.5 unk1.0 | 0.845 | 0.822 | 0.300 | 0.000 | 0.000 | 0.300 |
| w1.5 unk2.0 | 0.807 | 0.778 | 0.400 | 0.073 | 0.000 | 0.300 |
| w1.5 unk4.0 | 0.736 | 0.686 | 0.400 | 0.084 | 0.044 | 0.400 |

**Four things, in order of how much they should change behaviour.**

**1. `--unknown-weight` does exactly what the design wants, on synthetic data.**
Raising it trades top-1 for precision monotonically — val top-1 falls 0.880 to
0.669 while val precision-1.000 recall climbs 0.000 to 0.098, and test(US) from
0.000 to 0.044. There is a real frontier and the knob moves along it. This is
the first time it has been swept.

**2. The shipped configuration is the right one, and now on evidence.**
`w1.0 unk1.0` gives the best real-speaker operating point in the sweep — top-1
0.700 at precision-1.000 recall 0.500 — and si-device chose it before this ran.

**3. Do not select the model on synthetic accuracy.** `w1.5 unk0.5` is the best
model in the table on *both* synthetic populations (0.890 val, 0.864 test) and
is **worse on the real speaker** than the shipped one on the metric that
matters. A selection rule of "highest synthetic accuracy" picks it and ships a
worse device. Synthetic accuracy is a sanity check, not a selection criterion.

**4. The real test set cannot separate these models, and that is the finding.**
Real recall is measured over **ten keywords**: 0.500 against 0.300 is two
utterances. Across the four unknown-weights, w1.0 beats w1.5 on real twice,
loses once and ties once — which is noise, not a ranking. So the sweep can
*rank the knob* on synthetic data and *cannot rank the models* on real data.
The corpus is large enough to tune against; the human test set is not, and no
amount of synthetic data fixes that.

### What this means for a workshop

**test(US) precision-1.000 recall is 0.000 for six of eight configurations.**
Those are unseen American synthetic voices against a large adversarial negative
population — the closest available proxy for strangers — and against thousands
of attackers there is essentially no threshold at which the model never once
fires wrongly.

Read that carefully, because the real-speaker 0.500 sits beside it and they are
not measuring the same thing: the real set has **12** negatives, the synthetic
sets have **thousands**. A precision-1.000 point is far harder to reach against
thousands. The synthetic number is the pessimistic bound and the real number is
the optimistic one, and **the truth for a room of strangers is somewhere between
them, with nothing currently measuring where.**

That is the argument for [the protocol above](#the-protocol-for-the-user-to-run-without-us),
stated numerically: five real speakers with twelve negatives each is ~60
negatives, which starts to discriminate between an operating point that is
genuinely clean and one that has met twelve easy words.

## Tuning against the wrong model

**A `.tmdl` is not numerically the `.tflite` it was converted from.** Measured
on the board by si-device, over the eight sample patches: **top-1 differs on
three of eight**, and where it differs the device is systematically *more*
confident. Patch 0 goes from class 18 at p=0.566 on the host to class 21 at
p=0.949 on the device.

This is **not** a port bug and not specific to this model. The same comparison
against emlearn's own MNIST model — where the right answers are known
independently — also diverges, by up to 0.047 in probability. MNIST's argmax
survives because its margins are around 0.9. Ours do not, because ours run 0.00
to 0.38. TinyMaix requantises in float and casts; TFLite uses fixed-point
rounding multipliers with saturation. Different arithmetic, different numbers.

**So every threshold and margin in this document was tuned against a model that
is not the one that ships.** That is the same shape as the cross-microphone
problem `docs/speech.md` is built around: evaluating one thing and deploying
another, with nothing in between to say they disagree.

What it does and does not invalidate. The rule is **whether both sides of the
comparison move together**, and an earlier draft of this file got the most
important case wrong:

- **CNN-internal comparisons stand.** Real speaker against synthetic voices, the
  per-voice accent table, architecture against architecture. Both sides are the
  same model under the same arithmetic, so a shift moves them together and the
  ranking holds.
- **CNN against DTW does *not* stand, because only one side moves.** DTW is
  untouched by TinyMaix — it is integer L1 over the same features either way —
  while the CNN's probabilities shift. So a CNN figure measured on the `.tflite`
  and set beside DTW's measured figure is not a like-for-like comparison, and
  the CNN's 0.500 should be treated as provisional against it.
- **The operating point does not stand.** Precision 1.000 at recall 0.500,
  threshold 0.598, is a property of the `.tflite` and has to be re-derived.

**Do not read the direction as a recall cost.** Two of the three disagreements
went from a keyword to `unknown`, and at a *fixed* threshold that would cost
recall — but **the eight patches carry no ground truth.** Their filenames encode
what an earlier model predicted, not what was said. Where the host said a
keyword and the device said `unknown`, if the true label was `unknown` then the
**device was the more accurate of the two**, and nothing available says which.
What is measured is the direction of *disagreement*, not a direction of error.

### Host TinyMaix: built, and it reproduces the device

**This is done, and it is the thing that unblocks host-side tuning.** TinyMaix's
stock `include/tm_port.h` already carries emlearn's exact configuration —
`TM_ARCH_CPU`, `TM_OPT0`, `TM_MDL_TYPE = TM_MDL_INT8`, `TM_FASTSCALE 0`,
`TM_MAX_KCSIZE 3*3*256` — so no porting was needed, only a shim:

```
clang -O2 -shared -fPIC -I TinyMaix/include \
    TinyMaix/src/tm_model.c TinyMaix/src/tm_layers.c shim.c -o libtmhost.dylib
```

The shim mirrors `mod_cnn_run`: uint8 in, `TMPP_UINT2INT`, float out, driven
from Python by `ctypes`.

**One trap, and it fails the way everything in this project fails.** `tm_load`
fills the input `tm_mat_t` with the model's own dims and the upstream example
*reuses that same mat* as `tm_preprocess`'s output. Pass a fresh uninitialised
one and the dims are garbage — which presents as a **constant output for every
input**, not as noise. Same shape as the batch-norm bug: a plausible-looking
number that is the same plausible-looking number every time.

**It reproduces the device's decisions.** Against `si_real` and the eight
patches si-device ran on the board:

| | top-1 disagreements vs host TFLite | on which patches |
| --- | --- | --- |
| **device (si-device)** | 3 of 8 | 0, 6, 7 |
| **host TinyMaix (this build)** | **3 of 8** | **0, 6, 7** |

Same count, same patches, same predicted classes. Output probabilities land
within 2 to 8 of the 256 quantisation levels of the device's — close but not
bit-identical, and `-ffp-contract=off` does not close it, so the residue is
likely genuine ARM-versus-host float rounding in the requantisation. **Tune on
host TinyMaix; verify on the board any threshold sitting within ~0.03 of a
decision boundary.**

### What causes the divergence: two named things, and they are not all of it

The line is `arch_cpu.h`:

```c
outp[i] = (mtype_t)(sumf*out_s_inv + out_zp);
```

A bare C cast. It **truncates toward zero** where TFLite rounds to nearest, and
it **wraps** where TFLite saturates. The source even carries the rounding
variant commented out beside it.

Both were tested by building patched variants and measuring the divergence from
TFLite over the eight patches:

| variant | mean abs prob difference | max | top-1 agreement |
| --- | --- | --- | --- |
| stock | 0.01738 | 0.07013 | 5 / 8 |
| + round to nearest | 0.01447 | 0.06925 | 5 / 8 |
| + saturate to int8 | 0.01332 | 0.04936 | 5 / 8 |
| **both** | **0.01021** | **0.04457** | **5 / 8** |

**Neither is the whole story and fixing both does not help the decision.**
Rounding accounts for about 17% of the divergence and saturation about 23%;
together 41%, leaving the majority in the structural difference — TinyMaix
requantises in float per output element, TFLite uses fixed-point multipliers
with defined rounding. And **top-1 agreement does not move at all**: 5 of 8 in
every variant.

So patching TinyMaix is not a route to making TFLite-tuned thresholds valid on
the device. The route is to tune against TinyMaix, which the host build now
makes cheap.

### It is scatter, not a shift, and that is the real argument for the host build

On the MNIST control the deltas run **both ways** — `+0.0157` on digit 2,
`-0.0466` on digit 4. The two failure shapes have different remedies:

- A **monotone shift** moves the operating point along the precision/recall
  frontier. Re-tuning the threshold recovers it and the frontier is unchanged.
- **Scatter blurs the frontier itself**, because it degrades the separation
  between the classes the threshold exists to divide.

We have the second. So re-tuning under TinyMaix is **necessary but possibly not
sufficient**: the achievable curve may be genuinely worse than the one measured
under TFLite, not merely differently placed on it. That is a better reason for
the host build than "the threshold moved".

**What survives regardless.** DTW's half of the comparison is measured and firm:
it has **no threshold at all** at which it fires without a false positive, in
four configurations, and no arithmetic drift on the CNN's side can give DTW an
operating point it does not have. So "one of these can decline to answer and the
other cannot" holds — with one clause that costs nothing and is exactly true:
**the CNN clears precision 1.000 at non-zero recall under the arithmetic it was
measured on, and is expected but not yet confirmed to clear it under the
arithmetic it runs on.** The **size** of its advantage is not established and
should not be quoted until it is re-measured under TinyMaix.

**And the cost half is not in question.** 66.6 ms against DTW's 616-672 ms is
not a margin arithmetic drift can close.

The bias is at least in the safe direction — on all three disagreements the
device moved **toward `unknown`**, which for this toy means more deflections and
fewer confident misfires, exactly the trade `docs/speech-design.md` asks for.
But that is eight patches from a dry-run model. It is a hope, not a measurement.

**The fix, and it unblocks all host-side evaluation:** build TinyMaix for the
host from the same sources emlearn vendors (`TM_ARCH_CPU`, `TM_OPT0`,
`TM_MDL_TYPE = TM_MDL_INT8`, `TM_FASTSCALE 0` — the configuration in emlearn's
`src/tinymaix_cnn/int8/tm_port.h`), and sweep the gates against that. A couple
of hours of C build work. Failing that, sweep on the board: at 66.6 ms per
inference a 500-utterance sweep is about 35 seconds of board time per operating
point.

## The endpointer looked broken, and was being fed a chirp

This section is kept in the order it happened, because the wrong diagnosis was
reached twice on good evidence and the shape of that is worth more than the
conclusion.

**The symptom.** Ten utterances captured through the board — `src/vad.py` found
no word at all in **6 of them**, and the four it did find came back at 150-310 ms
against 300-640 ms for the same words synthetically: catching the vowel and
losing the onset, which is exactly what `docs/speech.md` warns stops separating
FATHER from MOTHER.

**The mechanism, and this part survived.** Writing `IMN` for the background
estimate (first 10 frames) and `IMX` for the loudest frame, `docs/speech.md`
gives `ITL = min(3*(IMX-IMN)/100 + IMN, 4*IMN)` and `ITU = 5*ITL`. The `4*IMN`
branch only binds above `IMX/IMN = 101`, which nothing approaches, so in
practice `ITU = 0.15*IMX + 4.85*IMN`. Speech is found only if `IMX > ITU`:

> **The endpointer requires the loudest frame to exceed
> `4.85 / 0.85 = 5.71` times the background.**

That predicted cutoff separated the ten takes **perfectly** — rejected at
3.6, 3.7, 4.4, 4.9, 5.5, 5.7 and found at 6.3, 7.1, 7.9, 8.6, with `father_01`
sitting exactly on the boundary and rejected.

**Two wrong conclusions drawn from it.** First, that the cause was clipping:
refuted by `dream_01`, which had zero clipped samples and was rejected anyway.
Then, that the cause was a loud room and the fix was to lower the capture gain:
refuted by arithmetic — `IMX/IMN` is **gain-invariant**, and simulating a 6 dB
reduction across all ten takes returned every ratio identical to three figures.
That left "the board's noise floor is 24-28 dB above the ES8311's own -44 dBFS,
so it is the room", and a recommendation to change `ITU`'s multiplier.

**All of which was measuring a bug.** `Recorder.chirp()` played a 300 ms
activation tone and waited on `play_finished()` — which reports that the DMA has
drained into the PIO FIFO, not that the speaker has stopped. Capture opened
while the tone was still sounding, so **every recording began with the tail of a
chirp**, landing precisely on the frames `IMN` is estimated from. A contaminated
background collapses `IMX/IMN`, which is the ratio the algebra identifies. The
mechanism was right; the thing inflating it was not the room.

Found and fixed by the lead in parallel with the analysis
(`CHIRP_SETTLE_MS = 140` in `src/listen.py`, timed from the start of playback
rather than from the DMA finishing). Their evidence, discarding the first N ms
of 22 pre-fix takes:

| skip | endpoints successfully |
| --- | --- |
| 0 ms | 8 / 22 |
| **300 ms** | **22 / 22** |
| 400 ms | 21 / 22 |
| 500 ms | 14 / 22 |

**Re-measured here on ten post-fix captures, independently:**

| | before | after |
| --- | --- | --- |
| endpointed | 4 / 10 | **10 / 10** |
| `IMX/IMN` | 3.6 – 8.6 | **22.6 – 49.3** |
| `IMN` (background) | 280k – 390k | **73k – 90k** |
| endpointed length | 150 – 310 ms | **250 – 440 ms** |

The background fell about fourfold and the ratio rose four to sixfold. **`ITU`
is nowhere near binding on a clean capture** — the margin is now 4x to 9x the
requirement — so do not touch it. The 5.71x rule has still never been tested
against this microphone; every measurement that appeared to test it was taken
through the bug.

### What this cost, and the rule that would have saved it

Two confident diagnoses, both from correct measurements, both wrong, because
every number came through a fault upstream of the thing being measured. The
algebra was right the whole way and pointed at `IMN` from the start; what was
missing was asking **why `IMN` was large** rather than assuming the answer.

`CLAUDE.md`'s standing rule is that "the code ran" never proves the hardware
did anything. The version of it that applies here: **a measurement of a signal
is not a measurement of its source.** A background estimate is a number about
the first 100 ms of a buffer, not about the room, and the two only coincide when
nothing else is in that buffer.

### The real-speaker result

`tools/si_eval.py build/si_real.tflite corpus-tts/manifest.json --takes ...`,
int8 model, 22 real utterances (10 keywords, 12 negatives):

| | top-1 | precision 1.000 at recall | at 95% precision |
| --- | --- | --- | --- |
| CNN, synthetic held-out voices (2) | 0.626 | 0.013 | 0.013 |
| **CNN, real speaker** | **0.700** | **0.500** | 0.500 |
| DTW control, same real speaker *(best of four rosters)* | 0.700 | 0.000 | — |
| DTW control, same real speaker *(no American voice, superseded)* | 0.300 | 0.000 | — |
| DTW control, held-out synthetic voices *(dry-run)* | 0.753 | 0.233 | 0.301 |
| DTW, same speaker (`docs/speech.md`) | — | 0.966 | — |

At threshold 0.598 the model fires on MOTHER, FATHER, HAPPY, SORRY and one of
the two SAD-class words, is silent on the other five keywords, and is **silent
on all twelve negatives** — including the near-rime attackers *other*,
*another*, *wonder*, *mothers*, *brothers* that set the DTW threshold, and the
retired *know*, *want*, *need*. Zero false fires of either kind.

**Evaluate against keywords and negatives together, or the number flatters.**
Pointing `--takes` at the ten keywords alone finds a "better" operating point —
threshold 0.469 at recall 0.700, apparently also with precision 1.000. It is
the same model on a test that omits the question: with no negatives present,
nothing can fire on an ordinary word, so a lower threshold survives. Restore
the twelve negatives at that threshold and one of them fires; precision is
0.875, not 1.000.

The tell is that the flattering result comes with a **lower** threshold. A
better model clears a higher bar; a weaker test lowers the bar. **0.500 at
threshold 0.598 is the number to quote.**

**The synthetic and real precision figures are not comparable, and the
difference is not the speaker.** The synthetic set has 1386 must-stay-silent
utterances chosen adversarially against 621 keywords; the real set has 12 against
10. A precision-1.000 operating point is far harder to reach against 1386
attackers than against 12, so the 0.013 is a much harsher test than the 0.500,
not evidence that synthetic voices are harder to recognise. Read the **top-1**
column for the speaker comparison and the **precision** column only within a row.

**Why the real speaker beats held-out synthetic voices** is worth stating rather
than celebrating: the two held-out voices are Daniel and Samantha, one family
each, and with n=2 the synthetic figure has an error bar of its own. The honest
reading is that the real-speaker result is *not worse*, which is the surprise —
`docs/speech-design.md` predicted a large fall and the brief for this work said
to expect one.

### Accent was real for DTW's top-1 and irrelevant to the decision

The first roster had **no American voice in training** while the speaker is
American, so every early number risked measuring an accent gap and calling it a
speaker gap. The roster was rebuilt with American voices in train (Bruce,
Samantha, Joelle, Noelle) and an American of each gender held out in test
(Allison, Nathan, both Enhanced). Re-running the DTW control — the cheapest read,
since it needs no training — against the same 22 real utterances:

| template voices | top-1 | precision 1.000 at recall |
| --- | --- | --- |
| 3, no American in roster *(old)* | 0.300 | 0.000 |
| 3, American-inclusive | 0.600 | 0.000 |
| **5, incl. Samantha + Joelle** | **0.700** | **0.000** |
| 8 voices x 2 takes (336 templates) | 0.500 | 0.000 |

**Accent was worth a lot to DTW's raw discrimination** — top-1 more than
doubled, 0.300 to 0.700 — and **worth nothing at all to the decision**, because
in every configuration there is still no threshold at which DTW fires without a
false positive. Its distances simply do not separate in-vocabulary from
out-of-vocabulary across speakers, however good its argmax gets. Note also that
piling on templates makes it worse, not better: 336 templates score 0.500
against 102 templates' 0.700, because every extra template is another chance for
an out-of-vocabulary word to land near something.

That sharpens the comparison rather than softening it:

| | top-1 | precision 1.000 at recall |
| --- | --- | --- |
| DTW, best configuration | 0.700 | **0.000** |
| CNN | 0.700 | **0.500** |

**Identical top-1, and only one of them is usable.** The difference is not
recognition, it is *rejection* — which is precisely what `docs/speech.md` argues
is the load-bearing property of this whole design, and precisely what a trained
`unknown` class provides and a nearest-neighbour distance does not.

#### And it was not limiting the CNN either, per held-out voice

On the **first** roster, before American voices were installed, the held-out
voices already spanned both cases, so the CNN's exposure to the confound was
measurable without waiting for a rebuild:

| held-out voice | accent | that accent in training? | CNN top-1 |
| --- | --- | --- | --- |
| **Samantha** | **en_US** | **no** | **0.668** |
| Tara | en_IN | yes | 0.653 |
| Daniel | en_GB | yes | 0.588 |
| Karen | en_AU | no | 0.562 |

**The single American voice scores highest of the four, and the accent that was
in training scores lowest.** Seen accents span 0.588-0.653; unseen accents span
0.562-0.668. Whether an accent appeared in training does not order this table,
so accent is not what is limiting the CNN — and the real speaker's 0.700 sits
right beside Samantha's 0.668, which is the comparison that matters: an American
voice, an accent absent from training, scoring the same as the American human.

**n=1 voice per accent.** This rules out accent as the *dominant* effect for
the CNN; it does not measure an accent penalty precisely.

**For DTW it did not rule accent out — and the re-run settled it.** A template
distance is far more sensitive to phonetic realisation than a convolutional net
pooled over time, so the 0.300 was suspected of being partly an accent gap. It
was: with American voices in the roster DTW's top-1 more than doubled, to 0.700.
Its precision-1.000 recall stayed at 0.000 in every configuration. So accent was
worth a great deal to DTW's discrimination and nothing at all to the decision --
the option was never blocked by top-1.

### What is still true

- **si-corpus's SNR result stands and is independent of this bug.** The same
  endpointer fails on **49% of utterances at 8 dB SNR and 83% at 6 dB**, against
  0% at 11 dB and above. The margin is thin even when the capture is clean, and
  `endpoint_ms: null` remains a real population in the corpus.
- **The clipping is fixed at source, and the corpus gap it revealed is not.**
  `MIC_GAIN` was 3, at which every real utterance pinned full scale — 2415
  saturated samples across ten keywords, in runs of 2-3, which is a waveform
  riding the rail at its peaks rather than isolated glitches. At gain 1 the
  peaks are 6723-23008 and **zero samples clip**. Lowering it was safe precisely
  because `IMX/IMN` is gain-invariant, so it could not cost the endpointer
  anything — and it did not: zero rejections either way.

  What survives is that **saturation appears nowhere in the training corpus**,
  which draws gain and tilt but never clipping. This particular microphone no
  longer produces it, but a louder speaker or a closer one will, and the corpus
  has no example of it. Still worth an augmentation axis; no longer urgent.
- **The feature dynamic-range gap is fully explained: it was the dry-run
  corpus, not the signal chain.** Standard deviation of the int8 patches:

  | set | n | std | p01 | p99 | cells at -128 |
  | --- | --- | --- | --- | --- | --- |
  | real, chirp-contaminated | 8 | 11.07 | -43 | 31 | 0.00% |
  | real, clipped (mic gain 3) | 22 | 17.91 | -51 | 50 | 0.00% |
  | **real, clean (mic gain 1)** | 22 | **16.88** | -53 | 50 | 0.00% |
  | **synthetic, real corpus (with noise)** | 800 | **17.48** | -54 | 49 | 0.00% |
  | synthetic, dry-run (no noise) | 800 | 44.12 | -128 | 91 | 3.77% |

  **16.88 against 17.48 — the generated corpus and real speech are
  indistinguishable.** The whole ~2.6x gap was an artefact of comparing against
  a corpus with no additive noise, exactly as predicted. Removing the clipping
  changed nothing (17.91 against 16.88), and the chirp was worth about six
  points; neither was the cause.

  The mechanism is the -128 clamp column. `say` renders true digital silence in
  the lead and tail, which after per-band mean subtraction is a large negative
  excursion — 3.77% of dry-run cells. Mix in a noise floor, as the real corpus
  does, and it vanishes to 0.00%, matching what a microphone produces. **This is
  the strongest single argument for si-corpus's noise augmentation**: without
  it the model would have trained on a contrast distribution the device can
  never deliver.
## Labels, used as [speech-design.md](speech-design.md) uses them

| Label | Means |
| --- | --- |
| *dry-run* | Measured on a **throwaway 10-voice corpus I generated to prove the harness**. Not a result. Never quote these as findings. |
| *real corpus* | Measured on `corpus-tts/`, the voice-disjoint corpus with a channel model. ~~There are none of these yet — the roster is frozen but the audio was never generated.~~ **Superseded: the audio was generated and this is now most of the document.** 16030 utterances over 8 training voices (`456ccf8`); see [the result](#the-result-in-three-numbers). |
| *verified* | Checked by running the thing, on the host |
| *unknown* | Not established — see [What is still unknown](#what-is-still-unknown) |

## Why synthetic voices at all

Training data is the whole difficulty. Google Speech Commands is 105,829
utterances from 2,618 speakers and overlaps this vocabulary by **two words**
(YES, and NO which has since been dropped) — no MOTHER, no FATHER, no DREAM, no
SAD. Recording real people costs hours each and we would want ten or more.
macOS `say` ships ~184 voices for free. That is the whole hypothesis.

Frame any model result against the DTW control below rather than against the
shipped 0.966: **a speaker-independent recogniser that reaches half the shipped
recall is doing well, not badly.**

## What the model sees, and why it is not a second front end

`tools/si_features.py`. Endpointing is `src/vad.py` through `tools/vad.py` —
the device's own. Features are `mfcc.logmel_q8()`, an accessor added to
`tools/mfcc.py` that returns **the 26 Q8 log2 mel values the existing front end
already computes on its way to every cepstrum**, taken from `work[3]`
immediately before the DCT.

A convolutional net wants the filterbank, not the cepstrum: neighbouring mel
bins are correlated in exactly the way a kernel exploits, and the DCT exists to
destroy that correlation — right for DTW's L1 distance, wrong here.

**Nothing new has to be verified on hardware.** `python3 tools/mfcc.py
--selftest` checks the accessor two ways, both *verified*: applying the DCT to
what it returns reproduces the raw cepstra exactly, and it matches the `mel`
entry of all five cases in `src/speech_fixtures.py` — the array the viper port
is already required to reproduce bit for bit — **5 of 5**. The device side is a
tap on an array it is already filling.

### The one new stage: normalisation to int8

```
m[i]    = trunc_toward_zero(sum over t of x[t][i], n_frames)     per band
y[t][i] = clamp((x[t][i] - m[i] + (1 << (S-1))) >> S, -128, 127)  S = 4
```

Per-**band** mean subtraction, which is the log-mel analogue of the cepstral
mean normalisation the DTW path already does, and there for the same reason: a
fixed channel colouration is additive in the log domain, so subtracting each
band's own mean removes it exactly.

**Measured consequence, and it redirected the corpus effort.** If per-band mean
subtraction removes any fixed linear filter, then gain and spectral tilt
augmentation are a no-op for these features. Checked on one utterance
("computer", Samantha, 150 wpm), patch values spanning -128..97:

| transform | mean abs diff | max |
| --- | --- | --- |
| gain -6 dB | 0.68 LSB | 15 |
| gain +3 dB | 0.32 LSB | 9 |
| tilt +0.15 | 0.34 LSB | 8 |
| tilt -0.15 | 0.35 LSB | 14 |

A fifth of a decibel — quantisation and the endpointer landing a frame
differently, not the transform surviving. Two of `say_corpus.py`'s three channel
axes therefore buy this path nothing. What *does* survive is **additive noise**,
**reverberation**, **speaking rate**, and far above all of them **more voices**.
The corpus generator was redirected accordingly and now mixes real background
noise at controlled SNR.

### Sizes, chosen by measurement

`N_FRAMES = 80`, `N_BANDS = 26`, `INPUT_SHIFT = 4`.

Endpointed lengths over 880 `say` utterances of all 22 forms (*dry-run*): min
35, median 55, p95 79, max 130 frames.

| N_FRAMES | centre-cropped |
| --- | --- |
| 64 | 23.3% |
| **80** | **4.3%** |
| 96 | 1.7% |
| 112 | 0.8% |

80 is where the curve flattens; 96 buys 2.6 points for 20% more
multiply-accumulates, and what is still cropped at 80 is the old formant voices,
on which the endpointer runs long, not genuinely long words.

**This number is due a re-derivation and is probably a little too large.** The
probe included the singing voices, which the frozen roster drops, and covered
~11 distinct voices rather than the 20 names it appeared to, so the 130-frame
tail is largely gone.

Real speech does **not** pull hard the other way, and an earlier draft of this
file said it did. That claim rested on the pre-fix captures, where the
endpointer was truncating words it had barely found — 15-31 frames. Measured on
the ten post-fix takes, all of which endpoint, the range is **25-44 frames**,
which sits inside the synthetic distribution rather than a third of it. A
word-dependent duration is an endpointer working; the constant one is the
failure `docs/speech.md` warns about, and that is what the old numbers were.

So 80 frames is generous but not absurd, and the case for shrinking it is now
about compute rather than about padding drowning the signal. Re-measure on the
natural tier before re-picking — si-corpus notes the 8 natural voices build in
about 15 minutes, and that is the population this choice has to fit.

Note `docs/speech.md`'s "28..61 frames" is **one voice at 172-178 wpm**. Twenty
voices across a wider rate range are half again as long. Do not reuse that range
for anything else.

The shift trades resolution against clipping (150 utterances, *dry-run*):

| shift | 1 LSB | clipped |
| --- | --- | --- |
| 3 | 0.188 dB | 20.2% |
| **4** | **0.376 dB** | **2.3%** |
| 5 | 0.753 dB | 0.0% |

4, because what clips is a band more than 48 dB below its own mean over the
utterance — the noise floor inside a silent frame, not any part of the word — so
the clip acts as a spectral floor, which `docs/speech.md` measured neutral for
DTW. Shift 5 spends half the int8 range on a value nothing depends on.

## The model

`tools/si_train.py`. Two architectures, both shaped by what TinyMaix can run
rather than by taste, and both **inside every budget by a wide margin**:

| | params | MMAC | output elements | weights | `.tmdl` |
| --- | --- | --- | --- | --- | --- |
| `dscnn` w1.0 | 13142 | 1.106 | 45142 | 11.4 KB | 21.1 KB padded |
| `dscnn` w2.0 | 42134 | 3.359 | 90262 | 38.2 KB | ~52 KB padded |
| `plain` w1.0 | 41222 | 2.237 | **16662** | 39.8 KB | 41.5 KB |

Against DTW's measured 616-672 ms of matching, all of them are cheap. Size will
not decide this.

**Output element count is in that table because it, not MACs, tracks the
requantisation cost** both back ends pay per output element — a fixed charge
that MAC counts miss entirely. `plain` w1.0 has twice `dscnn` w1.0's MACs and a
third of its output elements, so the two orderings disagree.

**It does not flip this decision, and an earlier draft of this file said it
did.** Costed on the real counts (si-device's model, ~65 cycles per output
element against ~36 per MAC), requantisation is **18.9% of `dscnn` w1.0 on
TinyMaix and 6.9% on viper** — so it is largest on TinyMaix, not on the viper
fallback, and halving the MACs beats tripling the output elements at these
ratios. `dscnn` w1.0 wins on both back ends, on viper by nearly 2x
(~285 ms against `plain`'s ~544 ms, both *predicted*). **Build `dscnn`.**

Keep the column: at a different shape it does flip the answer (2.8x the output
elements for 0.72x the MACs, in si-device's earlier comparison). One unmeasured
caveat in `plain`'s favour — a depthwise 3x3 has a contiguous inner run of 3
where a dense conv has `kw*chi`, so depthwise amortises viper's loop overhead
worse than a flat cost-per-MAC assumes. That penalty would have to be about 2x
to close the gap.

Three training choices are about precision rather than accuracy:

- **`unknown` is a trained class**, not a threshold. A softmax over keywords
  alone cannot represent "none of these" and produces confident nonsense at
  exactly the moment it matters.
- **Label smoothing is off.** It improves accuracy and flattens the softmax, and
  a flattened softmax is what a confidence threshold cannot work with. The usual
  advice points the wrong way here.
- **`--unknown-weight`** multiplies the unknown class's training weight, trading
  recall for precision directly. Sweep it; do not assume it.

## Measurement

`tools/si_eval.py` reports three things separately and never averages them:
synthetic held-out **voices**, real speaker, and precision swept against a
threshold and a margin. The gates map onto DTW's:

| `tools/dtw.py` | here |
| --- | --- |
| best distance <= THRESHOLD | top-1 probability >= `thresh` |
| second-best - best >= MARGIN | top-1 - top-2 >= `margin` |
| (no equivalent) | top-1 class is not `unknown` |

`measure()` is deliberately the same counting rules as `dtw.measure`, including
the three-way truth — in-vocabulary, benign morphological variant, must stay
silent — so the two are comparable. The table is printed as *highest recall at
each precision* rather than one row per threshold: DTW's integer scores have
plateaus and collapse to a few dozen rows, softmax probabilities are all
distinct and print a thousand rows of noise.

### The validation set is small, and that caps what can be concluded

The frozen roster holds 43 installed English voices, but only **8 are
independent natural speakers** — the rest are 16 iOS voices that form a single
family and 13 MacinTalk formant/novelty voices. Holding out a fifth of the
*voices* would put a sheep noise in the validation set, so the split holds out
half the *natural tier* instead:

| split | voices |
| --- | --- |
| train | 33 (16 expressive, 13 novelty, 4 natural) |
| val | 2 — Daniel, Samantha |
| test | 2 — Karen, Tara |

**Treat val as n=2 and do not report architecture differences from it.** Two
voices cannot separate a one- or two-point difference between `dscnn` and
`plain`, or between widths, from noise. The honest uses of this corpus are the
large effects — does the approach work at all, does the DTW control beat the CNN
— and nothing finer. Every manifest row carries a `tier` tag, so a different
split can be tried without regenerating audio.

This is a ceiling on the *idea*, not on the corpus tooling: `say` does not
supply enough independent vocal tracts on this machine to validate a
speaker-independent model properly, and that is worth knowing before anyone
spends another session on it.

### The control that decides how to read a bad result

`tools/si_dtw_control.py` runs **the incumbent, unchanged, speaker-independently**
— templates from `say` voices in the training split, queries from held-out
voices. Without it, "the CNN did badly" has two explanations that look identical
and lead to opposite decisions: the model is too small, or synthetic voices do
not transfer at all.

**Measured, *dry-run*** — 66 templates from 3 voices, 300 queries from 3 unseen
voices:

| | top-1 | precision 1.000 at recall | 95% precision at recall |
| --- | --- | --- | --- |
| DTW, same speaker (`docs/speech.md`) | — | 0.966 | — |
| **DTW, held-out speakers** *(dry-run)* | **0.753** | **0.233** | 0.301 |
| CNN, 7 training voices, no noise *(dry-run)* | 0.490 | 0.076 | 0.121 |

Read the first two rows and stop. **Changing speaker costs DTW 73 points of
recall at zero false fires** — that is the size of the problem, measured rather
than assumed, and it is the bar the CNN has to clear. The third row is a
deliberately impoverished harness test and **is not a result**; the corpus it
used has 7 voices and no channel augmentation, against the real corpus's 23 and
a full noise model.

## A method that paid for itself in ninety seconds

**When a model disagrees, re-run a model whose answers are known
independently.** The `.tmdl`-against-`.tflite` divergence looked exactly like a
broken ARM port, and would have been filed as one. Re-running emlearn's MNIST
model — whose correct answers come from upstream, not from us — showed the same
drift there, which is what turned an anecdote about our model into a mechanism
about the runtime. si-device ran it expecting to find their own bug, and it
happened to be the one check that could tell a port fault from a runtime
difference.

This is the third member of a family this project keeps producing, and
`docs/speech.md` already holds one of the others: the stress fixture built to
probe FFT overflow turned out **milder** than ordinary speech, because
pre-emphasis saturation clamped the pathological input before the transform saw
it. Add the chirp bleed, where a background estimate measured a tone rather than
a room. All three are *the measurement was fine, the thing measured was not the
thing meant* — and the general form is that **a comparison is only as symmetric
as the thing you changed.**

## Four traps, all of which fail silently

Grouped here because they are the same family as the entries in
`.serena/memories/gotchas_that_cost_time.md`: every one of them produces a
plausible number rather than an error.

### 1. `say -v` renders an uninstalled voice as the default, without failing

In a throwaway corpus, `Samantha`, `Fiona`, `Tom` and `Alex` had the **same
MD5**. Only Samantha was installed. Ten of twenty names were phantoms.

If phantoms land on both sides of a voice-disjoint split, the
speaker-independent score becomes a voice matched against itself and reads
**high**. Caught only because the DTW control returned distances of **exactly
0**, which real audio never produces.

`si_features.check_distinct_voices()` hashes every WAV and **raises**; it runs at
the top of `si_train.py`, `si_eval.py` and `si_dtw_control.py`.

### 2. The voice split leaked through locale variants

`roster.json`'s first split put every val voice's same-named counterpart in
train — Flo, Grandma, Grandpa, Rocko, Sandy and Shelley, US in val and UK in
train. Six of eight val voices, and **17 of 34 of the roster's own recorded
`close_pairs` straddled a split**. Those are one synthetic voice with a
different accent setting, not two speakers. Group by voice *name* before
grouping by acoustic fingerprint.

### 3. Batch norm's default momentum makes a model that is constant on held-out data

The first dry-run model reported **0.70 training accuracy and a flat 0.055 top
probability — 1/18 — on every held-out utterance**. Keras defaults
`BatchNormalization(momentum=0.99)`, which assumes thousands of steps; this
corpus gives a few tens per epoch, so the inference-time moving statistics were
still near their initial values when training stopped. Training metrics use
batch statistics and look fine throughout, so nothing points at batch norm.

`BN_MOMENTUM = 0.9` took keyword accuracy on held-out voices from **0.000 to
0.490** with no other change.

### 4. The int8 input contract holds by luck unless it is asserted

The device hands TinyMaix raw bytes, and the wrapper computes `quantised =
uint8 - 128` while **ignoring the model's own input scale and zero point**. So
the model's input quantisation must come out at exactly `scale 1.0, zero 0`. It
does — but only because the patches span the full int8 range, which is a
property of the *data*: a corpus that never clips would silently change the
scale, the device would feed numbers the model was not trained on, and the model
would load, run, and be quietly wrong.

`si_train.to_int8_tflite` now reads the quantisation back and **raises** unless
it is 1.0 / 0.

## The device path — it runs

**Measured on the board by si-device.** `emlearn_cnn_int8` imports on
`armv7emsp` (12096 B of heap), emlearn's MNIST model classifies 10/10 digits,
and **`si_real` runs at 66.6 ms per inference** — 1.106 MMAC at 9.03
cycles/MAC. Against DTW's 616-672 ms of matching that is **about 10x cheaper
than the matcher it replaces**, and flat in class count where DTW is linear in
template count.

Memory is a non-issue: 43536 B resident, 64624 B peak, and with the capture
buffer, ELIZA rules, framebuffer *and* the 137 KB DTW template set all held at
once there is still 174 KB free and a 102 KB largest block, with no measurable
slowdown (3218 us under full load against 3205 us on an empty heap).

**The padding was necessary and is now proved, with a control.** Loading the
unpadded `.tmdl` and filling the rest of the heap with a pattern, one inference
**overwrote 1164 bytes outside its allocation and raised nothing.** The padded
file, same script, same input: **0 bytes.**

Two device-side facts worth carrying: the wrapper rejects `bytearray` and needs
`array.array('B', ...)`, which costs **2.06x** the bytes it holds because
MicroPython grows arrays by doubling — that is where the peak comes from. And
measure with `gc.mem_alloc()`, not `gc.mem_free()`: a `mem_free` delta
over-reported this model's resident cost by 97%, because free-list
fragmentation is not resident use.

### What was verified on the host

Keras -> int8 `.tflite` -> `.tmdl`, all *verified* by running it. Findings that
constrain the model, from si-device's reading of the runtime and from
conversion:

- **No MAX_POOL_2D, no AVERAGE_POOL_2D, no REDUCE_MAX.** Only `MEAN`, as the
  global-average head. Every downsample is a strided convolution. A global *max*
  pool crashes the converter — reproduced.
- **The dense layer must be per-tensor.** TinyMaix's `tml_fc` reads `ws[0]` and
  applies it to every logit while the converter writes all 22 scales. It
  converts cleanly and returns wrong numbers for 21 of 22 classes. The
  converter flag exists on **TF 2.20 and not on TF 2.13**, so: **export the
  `.tflite` from the 2.20 venv, convert to `.tmdl` in the 2.13 one.**
- **`out_deq` must be 1**, or TinyMaix leaves the output int8 while emlearn
  reads it as float32.
- **The `.tmdl` must be padded, and this is now confirmed rather than
  suspected.** emlearn sizes TinyMaix's activation scratch as the model file's
  length, which is a guess, and `tm_load` does not check it against the model's
  own `buf_size`. `dscnn` w1.0 is **15776 bytes of file against a 21120-byte
  buffer** — it would write 5.2 KB past a heap allocation, silently, on a heap
  that also holds the ELIZA rules and the capture buffer. Depthwise-separable
  networks are structurally the shape that triggers it: few weights, large
  intermediate tensors. `tools/tmdl_info.py --pad` appends zeros until the file
  covers the buffer; the trailing bytes are never read. The model in `build/`
  is padded and `tmdl_info.py` exits 0 on it.
- **The module is `emlearn_cnn_int8`, not `tinymaix_cnn`.** It was renamed at
  emlearn-micropython 0.6.0 and split at 0.8.0; the old path is a 404. It is
  still the TinyMaix wrapper and is published prebuilt for `armv7emsp`.
- **Single-layer mode is not reachable.** `sub_size` is hardcoded to 0 in the
  converter, so the 3.6 KB figure TinyMaix's own tooling prints is what that
  mode *would* need, not what any `.tmdl` will ask for. Plan on
  `2 x max(file, buf_size)` resident, plus ~12 KB for the module. **Padding
  costs RAM as well as flash**, because both buffers are sized to the file
  length: `dscnn` w1.0 goes from 31552 B resident / 47328 B peak unpadded to
  **42240 B resident / 63360 B peak** padded. That is the trade, and it is
  still nothing against 489 KB contiguous.

**Two virtual environments are needed and both are `.venv*`-gitignored:**

```
uv venv --python 3.11 .venv     && uv pip install --python .venv/bin/python "tensorflow>=2.16,<2.21"
uv venv --python 3.10 .venv-tm  && uv pip install --python .venv-tm/bin/python "tensorflow-macos==2.13.0" "numpy<2" pillow
git clone --depth 1 https://github.com/sipeed/TinyMaix.git
touch TinyMaix/tools/__init__.py       # tflite2tmdl.py uses a relative import
cd TinyMaix && python -m tools.tflite2tmdl in.tflite out.tmdl int8 1 80,26,1 22
```

## What is still unknown

- **Whether it works on people, rather than on this person.** *Answered for one
  speaker*, in one session, in one room, over 10 keywords and 12 negatives. A
  second speaker is what turns this into a claim about human speech.
- **Whether the 5.71x `IMX/IMN` requirement is the right design.** Derived here
  and still never tested against this microphone — every measurement that looked
  like a test was taken through the chirp bug. On clean captures it is not
  binding, so this is a question for a noisier room rather than a live problem.
- **What the model does with more than 4 training voices.** Only the natural
  tier was generated; the 16-voice expressive family is unrendered. This is the
  largest untested lever.
- ~~Whether `emlearn_cnn_int8` imports on this board.~~ **Answered: it does,
  and `si_real` runs at 66.6 ms per inference.** See
  [The device path](#the-device-path--it-runs).
- **How far TinyMaix's arithmetic moves the operating point.** Top-1 differs on
  3 of 8 patches against the host `.tflite`, always toward `unknown`. Whether
  that costs recall, buys precision, or roughly cancels is unmeasured, and it is
  the single biggest open question about the headline number.
- **Whether TinyMaix's unclamped int8 outputs explain that divergence.** Its
  `tm_postprocess_sum` casts where TFLite saturates; a host-side counter of how
  often the pre-cast value leaves [-128, 127] would say. Not built, and now
  more interesting than when it was first raised.

## Resuming — the rest of the list

[Start here](#start-here--what-to-run-in-order) has the commands and the first
priority. The remainder, in order of value:

2. **Build TinyMaix for the host and re-tune the gates against it.** Until that
   exists, every threshold here is tuned on arithmetic the device does not run
   — see [Tuning against the wrong model](#tuning-against-the-wrong-model). It
   is a couple of hours of C and it unblocks all host-side evaluation; without
   it the gates have to be swept on the board at 66.6 ms an inference.
3. **Generate the rest of the corpus.** Only the natural tier was ever built.
   The 16-voice expressive family and the novelty tier are ~2 hours more, and
   the model that produced the headline trained on **4 voices**. More training
   voices is the largest untested lever.
4. ~~**Retrain on the rebuilt roster and re-run the evaluation.**~~ **Done
   (`456ccf8`).** `build/si_am.*` is that model: 8 training voices with American
   English in train, 16030 utterances. Held-out synthetic keyword accuracy went
   0.626 → 0.850 and **the real speaker did not move at all** — top-1 0.700,
   recall 0.500 at precision 1.000, both unchanged. Kept on the list as a
   result rather than deleted, because "more `say` output improved only the
   half that was already working" is the reason item 1 is more real speech and
   not more synthesis. Note `si_am` has never been converted to `.tmdl`.
5. ~~**Get the model onto the board.** Whether `emlearn_cnn_int8` imports at
   all is still unanswered.~~ **Done (`511a8ee`).** It imports on `armv7emsp`
   at 12096 B of heap, classifies emlearn's ten MNIST digits 10/10, and runs
   `si_real` at 66.6 ms an inference. See [The device path](#the-device-path--it-runs).
   What that session opened in its place is item 2: the `.tmdl` and the
   `.tflite` disagree on 3 of 8 patches, so the operating point measured on the
   host is not the operating point the board has.
6. **Then tune for precision.** Sweep `--unknown-weight`, `--width` and
   `--arch`. The operating point matters more than the accuracy and none of
   these has been swept once against the real corpus.
7. **Add clipping to the channel model.** This microphone no longer saturates
   at `MIC_GAIN = 1`, but a louder or closer speaker will, and the corpus has
   no example of saturation.

## The decision, and what happened to DTW

**The decision rule, stated in advance and settled:** the CNN had to beat the
DTW control's precision-1.000 recall on real speech. It did — **0.500 against
0.000**, at identical top-1 — so the speaker-independent path is not an
experiment any more. **The CNN ships.**

Not because DTW recognises worse. Because it cannot decline to answer.

**DTW is parked, not deleted**, and the distinction matters:

- `tools/mfcc.py` and `tools/dtw.py` remain the **ground truth for the front
  end**. Every feature this project computes, the CNN's included, comes out of
  `mfcc.py`, and `src/speech_fixtures.py` pins it bit-for-bit against the device
  port. `test_spotter` still measures against them.
- `tools/si_dtw_control.py` remains the **baseline any future model is measured
  against**. A speaker-independent number with nothing to compare it to is not
  a result.
- Speaker-*dependent* DTW remains the **fallback** if the CNN fails on hardware
  in a way that cannot be fixed. It is the only configuration in this project
  measured to work at recall 0.966.

What is dead is specifically **DTW with synthetically-enrolled templates**: four
configurations, no usable operating point, on a real speaker.

### The product changed, and it changes what to optimise

This was scoped as "can the ball recognise its owner without enrolment". It is
now a **workshop build**: many speakers, none of them known in advance, no
enrolment for any of them. That inverts the evaluation. Every number in this
document is for **one** speaker who happens to be the person who recorded the
takes; a room full of strangers is a different and harder population, and
nothing here measures it.

The consequence for tuning is concrete. For one known speaker, recall is worth
chasing. For a workshop, **precision against strangers is the thing that breaks
a demo** — a device that misfires on overheard chatter is worse than one that is
hard of hearing, because a deflection is in character and a wrong answer is not.
That is the same argument `docs/speech-design.md` made for the original design,
and a crowd sharpens it.
