# Shipping models

`build/` is gitignored -- it is where `tools/si_train.py` writes -- so the model
that actually ships lives here instead. A recogniser that vanishes on a clean
checkout is not a recogniser.

## `si_real.tmdl`

The speaker-independent keyword CNN. 21120 bytes, **already `--pad`ded** (15776
before; see below). Trained by `si-model` on `corpus-tts/`; converted with
TinyMaix's `tflite2tmdl.py` at `out_deq = 1`.

**Provenance: the training data is entirely synthetic.** `corpus-tts/` is
16,030 utterances rendered by eight macOS `say` TTS voices through a channel
model (`tools/si_corpus.py`); no human recording is in the training set, so
these weights contain no one's voice. (An earlier revision of this file called
it "the real-speaker corpus" -- that was the project's internal shorthand for
the *realistic* corpus, the one with noise and a channel model, as opposed to
the noise-free dry-run corpus. Human takes were used only as a local,
uncommitted evaluation set.) Retraining on your own corpus is fully
reproducible: see `tools/si_train.py`'s docstring.

| | |
| --- | --- |
| input | 80 frames x 26 log-mel bands, uint8, NHWC |
| input quantisation | scale 1.0, zero point 0, so the device sends `int8 + 128` |
| output | 22 float32 -- `vocab.LABELS` in order, then `unknown` |
| cost | 1.106 MMAC, 66.6 ms measured on this board |
| heap | 43 KB resident, 64 KB peak while loading |

**Measured through the device path** (`si_patch` + TinyMaix, not the `.tflite`)
on 10 real takes and 12 real out-of-vocabulary takes: **precision 1.000, recall
0.600** at `THRESHOLD = 0.35`, `MARGIN = 2/256`. See `docs/cnn-on-device.md`.

## The padding is load-bearing

`si_real.tmdl` is 21120 bytes of which **5344 are trailing zeros**. That is not
waste. emlearn's wrapper sizes TinyMaix's activation scratch to the model file's
length and never checks it against the model's own `buf_size`, so an unpadded
15776-byte file writes past its heap allocation. Confirmed on this board: 1164
bytes overwritten by one inference, nothing raised.

Run `python3 tools/tmdl_info.py models/si_real.tmdl` before deploying anything
here. `tools/deploy.sh` refuses a model that fails it.

## `si_am` is not here

The other trained model covers the emotional words (dream, happy, sad, sorry)
rather than the family nouns. It has never been converted to a `.tmdl`. The
family nouns are what ELIZA gets the most out of echoing -- "DO YOU OFTEN THINK
OF MOTHER" is the line that makes the toy feel alive -- so `si_real` ships. If
that judgement is revisited, `CNN_MODEL` in `tools/deploy.sh` is the one line.
