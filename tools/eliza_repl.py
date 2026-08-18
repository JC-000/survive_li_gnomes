#!/usr/bin/env python3
"""Converse with src/eliza.py on the host, with no board attached.

    uv run tools/eliza_repl.py                 # full ELIZA, ordered input
    uv run tools/eliza_repl.py --bag           # as a keyword spotter sees it
    uv run tools/eliza_repl.py --bag --show    # ...and print the bag each turn
    uv run tools/eliza_repl.py --bag --vocab words.txt

`--bag` is the point of this tool. It takes the sentence you typed, throws away
every word outside the vocabulary and every trace of word order, and hands the
engine what a small keyword spotter would actually have produced. That is how to
judge whether the degraded version is worth building *before* anyone builds a
spotter for it -- the alternative is finding out after the hard part is done.

Type `:quit` to leave, `:vocab` to print the vocabulary.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import eliza  # noqa: E402


# A 40-word vocabulary, sized to what a small spotter might plausibly manage,
# and split the way the rule analysis argues it should be: a spotter that knows
# only trigger words can only ever produce canned lines, which is a fortune
# cookie. The nouns are what let it say something the user just said.
#
# Every word here is one the script actually reacts to -- a keyword, a member of
# one of its word classes, or a word it tags as a noun. Adding words the script
# has never heard of buys nothing.
# **Ordered by value per slot, best first.** A recogniser that can only manage
# 25 words should take the first 25 and stop; nothing later in the list is a
# prerequisite for anything earlier.
VOCABULARY = (
    # 1. Feeling words. Best value in the list: each one both triggers a rule
    #    and fills its slot, so a bag of exactly one of them still produces the
    #    strongest line in the script ("I AM SORRY TO HEAR YOU ARE SAD").
    #    Straight from the script's own (* SAD UNHAPPY DEPRESSED SICK) and
    #    (* HAPPY ELATED GLAD BETTER) classes -- words it has no class for, so
    #    ANGRY, AFRAID and LONELY however natural they sound, do nothing at all
    #    and are not worth a slot.
    "SAD", "UNHAPPY", "DEPRESSED", "SICK", "HAPPY", "GLAD", "BETTER",
    # 2. Family nouns. Tagged /FAMILY by the script, so they reach the family
    #    rules *and* get echoed by name. The only words that make the toy sound
    #    like it heard who you were talking about.
    "MOTHER", "FATHER", "SISTER", "BROTHER", "WIFE", "HUSBAND", "CHILDREN",
    # 3. Other nouns. The script never tagged these, so they get the generic
    #    possessive echo ("YOUR WORK") rather than the family rules -- still the
    #    user's own word coming back at them, which is the point.
    "WORK", "MONEY", "SLEEP", "DEATH", "LOVE", "SCHOOL", "FRIEND",
    # 4. Triggers. Each unlocks a bank of canned replies: topical, never
    #    specific. No alias forms -- MAYBE, ALIKE and MACHINE would each cost a
    #    slot to reach a rule PERHAPS, SAME and COMPUTER already reach.
    "YES", "NO", "SORRY", "PERHAPS", "ALWAYS", "BECAUSE", "WHY",
    "HELLO", "NAME", "REMEMBER", "SAME", "DREAM", "DREAMED",
    "COMPUTER", "EVERYONE", "EVERYBODY", "NOBODY",
    # 5. Weakest. Both are word-class members, but every template behind them
    #    needs a clause, so alone they deflect. Last to add, first to cut.
    "WANT", "NEED",
)

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
NOUNS = ("MOTHER", "FATHER", "SISTER", "BROTHER", "WIFE", "HUSBAND",
         "CHILDREN", "WORK", "MONEY", "SLEEP", "DEATH", "LOVE", "SCHOOL",
         "FRIEND")

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
PRIORITY = {
    "MY": 6,        # the family and possessive rules: these keep the noun
    "I": 2,         # the feeling rules: strongest alone, lossy alongside a noun
    "AM": -4,       # defers straight to WHAT, so it answers "WHY DO YOU ASK"
    "ARE": -4,      # same
    "YOU": -2,      # broad, and mostly needs a clause we do not have
}


def to_bag(text, vocabulary):
    """Simulate the spotter: keep recognised words, discard everything else.

    Sorted rather than shuffled so a session is reproducible. Sorting is not
    "the order they were said" -- that is exactly the point, and the engine's
    tie-breaking has to cope with an order that means nothing.
    """
    seen = []
    for word in eliza.normalise(text):
        word = word.strip(",.!?;")
        if word in vocabulary and word not in seen:
            seen.append(word)
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

    vocabulary = VOCABULARY
    nouns = NOUNS
    if args.vocab:
        with open(args.vocab) as handle:
            vocabulary = tuple(w.strip().upper() for w in handle if w.strip())
        # Without a declared noun tier, fall back to the script's own tagging.
        nouns = None

    doctor = eliza.Doctor(priority=PRIORITY if args.bag else None)
    dress = (lambda s: s) if args.plain else eliza.sentence_case

    if args.bag:
        print("-- bag mode: %d word vocabulary, order discarded" % len(vocabulary))
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
            bag = to_bag(line, vocabulary)
            if args.show:
                print("  [spotted: %s]" % (" ".join(bag) if bag else "nothing"))
            reply = doctor.respond_to_keywords(bag, nouns=nouns)
        else:
            reply = doctor.respond(line)

        print(dress(reply))

    return 0


if __name__ == "__main__":
    sys.exit(main())
