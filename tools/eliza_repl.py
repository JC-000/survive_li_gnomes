#!/usr/bin/env python3
"""Converse with src/eliza.py on the host, with no board attached.

    uv run tools/eliza_repl.py                 # full ELIZA, ordered input
    uv run tools/eliza_repl.py --bag           # as a keyword spotter sees it
    uv run tools/eliza_repl.py --bag --show    # ...and print the bag each turn
    uv run tools/eliza_repl.py --bag --vocab words.txt

`--bag` is the point of this tool. It takes the sentence you typed, throws away
every word outside the vocabulary and every trace of word order, and hands the
engine what the keyword spotter would actually have produced. That is how to
judge whether the degraded version is worth building *before* anyone builds a
spotter for it -- the alternative is finding out after the hard part is done.

**The vocabulary is `src/vocab.py`'s, not a list of its own**, so what this
shows off is what the board can say. It is deliberately smaller than it looks:
22 spoken forms in 21 classes, and the words a therapist hears most -- no, want,
need, why, because, remember -- are not in it. See the note above `SPOTTABLE`
for what happened the last time this file kept its own copy. `--vocab` takes a
word list if the question is what a *larger* vocabulary would buy.

Type `:quit` to leave, `:vocab` to print the vocabulary.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import eliza  # noqa: E402
import vocab  # noqa: E402


# --- The vocabulary, derived from src/vocab.py rather than restated ---------
#
# This tool exists to preview the device honestly, so it has to hear exactly
# what the device hears and no more. It used to carry its own hand-written
# 40-word list, which drifted: it was 40 words against the device's 21 classes,
# and among the 21 it could hear and the board could not were **NO, WANT and
# NEED** -- all three deliberately retired from vocab.py, NO because it is a
# homophone of KNOW and no acoustic matcher separates them -- plus SCHOOL and
# FRIEND, which were never in the vocabulary at all. A preview that flatters by
# 19 words is worse than no preview, because the replies it shows off are the
# ones the board will never produce.
#
# That is the same drift tools/enrol.py had and fixed the same way; the rule
# this project keeps arriving at is that a second copy of the vocabulary is
# always eventually a wrong copy. tools/test_eliza.py pins the relationship.
#
# The design reasoning that used to live here -- why the budget goes to
# echoable nouns rather than trigger words, why SAD/SICK and WANT/NEED share a
# class, why NO cost 82 points of recall -- is in src/vocab.py's header, which
# is where it belongs now that this file no longer decides anything.
#
# One claim from the old list did not survive contact and is recorded rather
# than deleted: it said ANGRY and AFRAID "do nothing at all and are not worth a
# slot", the script having no word class for them. Checked against the engine
# as it stands, both produce "I AM SORRY TO HEAR YOU ARE ANGRY" -- a real reply
# with the word echoed. Whatever was once true of them is not true now.

# Spoken form -> the bag word the device would emit for it. Two forms map to
# one word wherever the engine treats them identically: say "sick" to the board
# and the spotter returns the `sad` class, so the engine sees SAD. Modelling
# that here matters -- it is why confusing the pair costs nothing, and a
# preview that kept them separate would be testing a distinction the device
# does not make. This is exactly `vocab.ECHO.get(label, ...)` in
# talk.Conversation.reply, which is the line this stands in for.
SPOTTABLE = dict((form.upper(), entry[1])
                 for entry in vocab.VOCAB for form in entry[2])

VOCABULARY = tuple(sorted(SPOTTABLE))

# No function words, deliberately. MY, YOUR, I, YOU, AM and ARE were once here
# because the family rules hang off keyword MY and the feeling ones off I -- so
# a spotted noun could not reach its own rule without them. They are also the
# hardest words to recognise: one or two phonemes, unstressed, and run together
# with whatever follows. Asking the recogniser for its six least reliable words
# and then making every good reply depend on them was the wrong trade twice
# over. eliza._ASSUMED_IN_BAG supplies them instead, and the slots went to
# nouns. tools/test_eliza.py:TestNoFunctionWordsNeeded holds the line.

# The subset the engine may drop into a slot meant for a clause. Keeping this
# tight matters: echoing "YOUR YES" would be worse than saying nothing.
#
# vocab.py's own, not a copy of it -- this is the same tuple talk.py hands the
# engine (`Conversation._ensure`), so a noun echoed here is a noun the board
# will echo. The copy that used to live here had gained SCHOOL and FRIEND,
# which are in no vocabulary the spotter has ever been given.
NOUNS = vocab.NOUNS

# Extra rank, added to the script's own, for deciding which spotted keyword
# answers. A bag has no word order, so without this the winner is whichever
# keyword happens to sort first -- and the script leaves most of its keywords at
# rank 0 precisely because it expected order to settle it.
#
# The rule is: promote the keywords whose rules can quote something back, demote
# the ones that only defer. MY owns the family patterns and I owns the feeling
# ones, so between them they are almost every reply worth hearing. AM and ARE
# own nothing -- both immediately hand off to WHAT -- yet "AM" sorts ahead of
# "I", which is enough to turn "I am unhappy" into "Why do you ask".
# Values are chosen to sit *within* the script's own scale (1-50), not above it:
# nudging I to 6 put it over REMEMBER's 5, so "do you remember" started getting
# answered by the generic I rules instead. Promote against the rank-0 crowd,
# and leave the keywords the script bothered to rank alone.
# Since the function words left the vocabulary these no longer rank words the
# spotter heard -- they rank the keywords the engine *assumes*, which is where
# the decision now lives.
#
# MY above I is load-bearing and counterintuitive. The feeling rules produce the
# better sentence when a feeling is all that was heard, but they fill their slot
# from the emotion word and never mention the noun. Rank them first and "my
# brother is sick" and "my children are sick" both come back as "I AM SORRY TO
# HEAR YOU ARE SICK" -- the same reply for different utterances, and it tells
# the user they are sick when they said their brother was. The noun rules keep
# the subject, so the noun wins whenever there is one.
# See tools/test_eliza.py:TestNounBeatsEmotion.
#
# The table itself is vocab.PRIORITY, the one talk.py passes to eliza.Doctor.
# It happened to still agree with the copy that used to be here, which is luck
# rather than maintenance -- the vocabulary copy beside it had drifted by 19
# words. The commentary stays because it records why these five numbers are
# these five numbers, which the data alone does not say.
PRIORITY = vocab.PRIORITY


def to_bag(text, spottable):
    """Simulate the spotter: keep recognised words, discard everything else.

    `spottable` maps a spoken form to the word the device would put in the bag,
    so "sick" comes back as SAD -- the spotter returns a *class*, not the word
    that was said, and two forms sharing a class is the whole reason confusing
    them is free. Returning the form instead would test a distinction the board
    cannot make.

    Sorted rather than shuffled so a session is reproducible. Sorting is not
    "the order they were said" -- that is exactly the point, and the engine's
    tie-breaking has to cope with an order that means nothing.
    """
    seen = []
    for word in eliza.normalise(text):
        word = word.strip(",.!?;")
        bagged = spottable.get(word)
        if bagged is not None and bagged not in seen:
            seen.append(bagged)
    return sorted(seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", action="store_true",
                    help="feed the engine a bag of spotted keywords, not a sentence")
    ap.add_argument("--show", action="store_true",
                    help="print the bag the spotter would have produced")
    ap.add_argument("--vocab", help="file of recognised words, one per line")
    ap.add_argument("--plain", action="store_true",
                    help="leave replies in the script's upper case")
    args = ap.parse_args()

    spottable = SPOTTABLE
    nouns = NOUNS
    if args.vocab:
        # An explicit word list is one word per bag entry, with no classes:
        # nothing outside vocab.py has a class to belong to. This is the escape
        # hatch for asking "what would a bigger vocabulary buy" -- which is a
        # different question from "what will the board do", and the default is
        # the one that answers the second.
        with open(args.vocab) as handle:
            words = [w.strip().upper() for w in handle if w.strip()]
        spottable = dict((w, w) for w in words)
        # Without a declared noun tier, fall back to the script's own tagging.
        nouns = None
    vocabulary = tuple(sorted(spottable))

    doctor = eliza.Doctor(priority=PRIORITY if args.bag else None)
    dress = (lambda s: s) if args.plain else eliza.sentence_case

    if args.bag:
        print("-- bag mode: %d spoken form%s in %d classes, order discarded"
              % (len(vocabulary), "" if len(vocabulary) == 1 else "s",
                 len(set(spottable.values()))))
    print(dress(doctor.greet()))

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in (":quit", ":q"):
            break
        if line == ":vocab":
            print("  " + " ".join(sorted(vocabulary)))
            continue

        if args.bag:
            bag = to_bag(line, spottable)
            if args.show:
                print("  [spotted: %s]" % (" ".join(bag) if bag else "nothing"))
            reply = doctor.respond_to_keywords(bag, nouns=nouns)
        else:
            reply = doctor.respond(line)

        print(dress(reply))

    return 0


if __name__ == "__main__":
    sys.exit(main())
