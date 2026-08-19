"""What the keyword spotter listens for, and what ELIZA does with each hit.

Twenty-two spoken forms in twenty-one classes. The budget is spent on
**echoable nouns** rather than on trigger words, because a reply that says the
user's own word back -- "DO YOU OFTEN THINK OF MOTHER" -- is what makes the
thing feel alive, while a trigger word can only ever unlock a canned line, and
canned lines read as fortune cookies. Everything not spotted falls through to
ELIZA's deflections, which are in character; that is why the spotter is tuned
for precision and not for recall.

## Classes, not words

Two spoken words share a class when ELIZA would treat them identically. SICK
and SAD both match DOCTOR's own `(*SAD UNHAPPY DEPRESSED SICK)` word class, and
WANT and NEED both match `(*WANT NEED)`, so confusing one for the other costs
nothing at all. That turns the two nastiest acoustic collisions in the list into
non-events, and it is why they were kept rather than dropped.

The echoable nouns cannot be merged that way -- the echo is the whole point --
so their collisions are real and had to be measured instead. See
docs/speech.md for the confusion matrix.

## NO was in this list and had to come out

NO is /nou/. So is KNOW. They are homophones, not near-misses, and no acoustic
matcher separates them -- while "I don't know" and "you know" are among the
commonest things anyone says to a therapist. Measured on the evaluation corpus,
"know" matched the NO templates at a distance of 172 when the best genuine
in-vocabulary match in the whole set scored 143. There is no threshold between
them.

Keeping NO capped the entire vocabulary at 7% recall for zero false fires,
because one class's false alarms set the threshold for all twenty-three.
Removing it took that to 89.6% immediately, and to 99.0% once the margin gate
was added. One short word cost eighty-two points of recall everywhere else.

THOUGH and FELT attacked NO the same way. The lesson generalises: a single
short, unstressed, vowel-dominated token is the worst possible thing to put in
a small-vocabulary spotter, because English is full of function words that
sound like it. Long polysyllables (COMPUTER, CHILDREN) are nearly free.

"no" now falls through to a deflection, which is in character.

## `echo` is upper case because the rules are

It goes straight into a reply template, and the whole 1966 script is upper case
-- it was written for a teletype. Matching is case-sensitive, so a lower-case
echo here would silently fail to fill any slot.

It is not upper case for effect: `screen.py` renders replies through
`eliza.sentence_case()`, so what the user reads is "your mother", not "your
MOTHER". Nine lines of 16-pixel capitals is markedly harder to read than a
sentence, and a single shouted word mid-reply looks like a bug rather than
emphasis.
"""

# (label, echo text, spoken forms recorded as templates, kind)
#
# `kind` is what ELIZA does with a hit, and it is data rather than a comment
# because eliza.Doctor needs to be told which words are worth echoing -- see
# NOUNS below. "noun" is echoed back by name, "feeling" fills a slot in the
# script's own word class, "trigger" only unlocks a bank of canned lines.
VOCAB = (
    # --- echoable nouns ----------------------------------------------------
    # Twelve separate classes: the reply repeats the word, so a confusion here
    # is visible to the user and each needs its own identity.
    ("mother",   "MOTHER",   ("mother",),        "noun"),
    ("father",   "FATHER",   ("father",),        "noun"),
    ("sister",   "SISTER",   ("sister",),        "noun"),
    ("brother",  "BROTHER",  ("brother",),       "noun"),
    ("wife",     "WIFE",     ("wife",),          "noun"),
    ("husband",  "HUSBAND",  ("husband",),       "noun"),
    ("children", "CHILDREN", ("children",),      "noun"),
    ("work",     "WORK",     ("work",),          "noun"),
    ("money",    "MONEY",    ("money",),         "noun"),
    ("sleep",    "SLEEP",    ("sleep",),         "noun"),
    ("death",    "DEATH",    ("death",),         "noun"),
    ("love",     "LOVE",     ("love",),          "noun"),

    # --- emotional state ---------------------------------------------------
    # SAD/SICK and WANT/NEED are merged because DOCTOR merges them.
    ("sad",      "SAD",      ("sad", "sick"),    "feeling"),
    ("happy",    "HAPPY",    ("happy",),         "feeling"),
    ("angry",    "ANGRY",    ("angry",),         "feeling"),
    ("afraid",   "AFRAID",   ("afraid",),        "feeling"),
    # WANT / NEED was here and was removed. Not for space -- because its own
    # rule provably never fires. Every template behind DOCTOR's (* WANT NEED)
    # class needs an object ("WHY DO YOU WANT 4"), so WANT says nothing alone;
    # and whenever an object *is* spotted it is a noun, so the possessive rules
    # outrank it (PRIORITY, below) and answer instead. Checked exhaustively over
    # every bag of one to three vocabulary words containing it: zero firings.
    # tools/test_eliza.py:test_want_never_reaches_its_own_rule asserts that, and
    # is the thing to look at if you are considering putting it back -- if that
    # test ever fails, WANT has become useful and this comment is stale.
    #
    # It was also the pair most likely to be clipped on capture, both words
    # being short and ending in a low-energy stop, and dropping it was measured
    # acoustically neutral on the 343-file corpus (threshold 706 -> 703, recall
    # 0.979 -> 0.977, no false fires either way; the binding constraint is
    # "other" -> FATHER at 727 and is untouched).

    # --- triggers ----------------------------------------------------------
    ("yes",      "YES",      ("yes",),           "trigger"),
    ("dream",    "DREAM",    ("dream",),         "trigger"),
    ("computer", "COMPUTER", ("computer",),      "trigger"),
    ("always",   "ALWAYS",   ("always",),        "trigger"),
    ("sorry",    "SORRY",    ("sorry",),         "trigger"),
)

LABELS = tuple(entry[0] for entry in VOCAB)
ECHO = dict((entry[0], entry[1]) for entry in VOCAB)
FORMS = tuple(form for entry in VOCAB for form in entry[2])

# Pass to eliza.Doctor.respond_to_keywords(spotted, nouns=vocab.NOUNS).
#
# Upper case, because it is compared against the echo text the spotter emits,
# not against the labels. Passing labels would silently match nothing and every
# echo would quietly disappear -- the failure looks like blandness, not an error.
#
# Without this the engine falls back to its own default_nouns(), which is the
# 1966 script's family list: it knows MOTHER but not WORK, MONEY, SLEEP, DEATH
# or LOVE, so five of the twelve nouns would never be echoed.
NOUNS = tuple(entry[1] for entry in VOCAB if entry[3] == "noun")
FEELINGS = tuple(entry[1] for entry in VOCAB if entry[3] == "feeling")
TRIGGERS = tuple(entry[1] for entry in VOCAB if entry[3] == "trigger")

# Pass to eliza.Doctor(priority=vocab.PRIORITY).
#
# These rank the keywords the engine *assumes* rather than ones the spotter
# heard: a bag of {MOTHER} reaches DOCTOR's family rules only because the engine
# supplies the "MY" nobody can recognise reliably. Without a table the script's
# own ranks decide, and it leaves most keywords at 0 because it expects word
# order to break the tie -- which a bag does not have.
#
# MY above I is the load-bearing line, and it is counterintuitive. The feeling
# rules give the better sentence when a feeling is all that was heard, but they
# fill their slot from the emotion word and never mention the noun. Rank them
# first and "my brother is sick" and "my children are sick" both come back as
# "I AM SORRY TO HEAR YOU ARE SICK" -- one reply for two utterances, and it
# tells the user *they* are sick when they said their brother was.
PRIORITY = {
    "MY": 6,        # the family and possessive rules: these keep the noun
    "I": 2,         # the feeling rules: strongest alone, lossy beside a noun
    "AM": -4,       # defers straight to WHAT, answering "WHY DO YOU ASK"
    "ARE": -4,      # same
    "YOU": -2,      # broad, and mostly needs a clause a bag cannot supply
}


def label_of(form):
    for entry in VOCAB:
        if form in entry[2]:
            return entry[0]
    return None
