#!/usr/bin/env python3
"""Convert Weizenbaum's 1966 DOCTOR script into src/eliza_rules.py.

Runs on the host, not the device. The output is committed, so this only has to
be re-run when the source script or the classification below changes -- but it
is kept, rather than the generated file being a one-off blob, because the
interesting decisions live *here*: which template gets which kind, and what
counts as a slot a single noun can fill.

    uv run tools/make_eliza_rules.py                 # fetches the script
    uv run tools/make_eliza_rules.py path/to/script.txt

Source: Anthony Hay's verbatim transcription of the appendix to Weizenbaum's
January 1966 CACM paper, which is the only published form of the script.
https://github.com/anthay/ELIZA/blob/master/scripts/ELIZA-script-DOCTOR-original-1966-CACM-appendix.txt

The script is a Lisp-ish list structure. A keyword entry looks like

    (REMEMBER 5
        ((0 YOU REMEMBER 0)                     <- decomposition
            (DO YOU OFTEN THINK OF 4)           <- reassembly templates
            (WHAT ELSE DO YOU REMEMBER))
        ((0 DO I REMEMBER 0)
            (=WHAT)))                           <- a goto, not a reply

with a handful of other constructs documented at the classification functions
below. Digits inside a reassembly are 1-based references to the components the
decomposition split the input into.
"""

import re
import sys
import urllib.request

SCRIPT_URL = (
    "https://raw.githubusercontent.com/anthay/ELIZA/master/scripts/"
    "ELIZA-script-DOCTOR-original-1966-CACM-appendix.txt"
)

OUT_PATH = "src/eliza_rules.py"

# Reply kinds. These are the whole point of generating the file rather than
# hand-writing it: a keyword spotter cannot recover an arbitrary run of the
# user's words, so the device build ships only the kinds it can actually fill.
# See the header written into the generated module.
CANNED = "C"
LITERAL = "L"
NOUN = "N"
PHRASE = "T"

# Control kinds, which produce no text of their own.
GOTO = "G"
NEWKEY = "K"
PRE = "P"

# A slot is NOUN-fillable only where a bare noun is grammatical on its own --
# "WHAT ABOUT 5" reads fine as "WHAT ABOUT YOUR MOTHER". Every entry here buys
# echo capacity, which is the scarcest thing the degraded path has.
NOUN_CARRIERS = (
    # Prepositions and possessives.
    "OF", "ABOUT", "YOUR", "MY",
    # Transitive verbs, which take a bare noun object just as readily:
    # "WHY DO YOU WANT MONEY", "SUPPOSE YOU GOT MONEY SOON".
    "GOT", "WANT", "WANTING", "REMEMBER", "FORGET",
)

# Deliberately NOT carriers, though they precede a slot: WAS and the other
# copulas take a predicate, not an object, so filling them with a noun gives
# "PERHAPS I WAS MOTHER". THAT wants a clause, and yields "DO YOU THINK ITS
# LIKELY THAT MOTHER". Widening this list is how the degraded version starts
# sounding broken, so widen it only by reading the results out loud.

# Amendments to the script's own DLIST tags. Kept separate and short so the
# transcription stays honest about what is Weizenbaum's and what is ours.
#
# The 1966 script tags WIFE as /FAMILY but has no entry for HUSBAND at all,
# which is a period artefact rather than a design decision -- the whole family
# rule fires for "my wife" and falls flat for "my husband". Everything else in
# its /FAMILY list is symmetrical, so this restores the symmetry.
EXTRA_TAGS = {
    "HUSBAND": ("FAMILY",),
}

# Extra members for the script's (* ...) word classes, keyed by a word already
# in the class we want to extend.
#
# ANGRY and AFRAID are among the first things anyone says to a therapist, and
# the 1966 class has neither -- so a spotter that recognises them perfectly
# still gets a deflection, which wastes two of a twenty-three class budget. The
# templates behind the class carry no assumption beyond "this is a bad feeling"
# ("I AM SORRY TO HEAR YOU ARE 5", "CAN YOU EXPLAIN WHAT MADE YOU 5"), so both
# words drop straight in and read correctly.
EXTRA_CLASS_MEMBERS = {
    "SAD": ("ANGRY", "AFRAID", "LONELY", "ANXIOUS"),
    "HAPPY": ("BETTER",),
}

# --- Mood, and why the script's replies get punctuation it never had ---------
#
# The 1966 script carries no terminal punctuation at all. On a teletype that
# costs nothing: replies scroll past in a transcript and the next prompt line
# delimits them. Alone on a 200x200 panel, one reply and nothing else, the same
# text reads as unfinished -- and the shortest ones read as broken. A user on
# the device reported "YOUR BROTHER" as "a sentence fragment and not a follow on
# question", which is exactly right about how it reads and exactly wrong about
# what it is: DOCTOR's bare echo is an invitation to go on, and on a teletype it
# looked like one.
#
# So every reply is tagged with a mood and given the punctuation that goes with
# it. Same trade as the I -> ME fix in eliza.fill: authenticity loses to
# legibility when the medium changed underneath the text.
#
# The tag is data rather than a rule in the display code because it has a second
# consumer. Spoken (see docs/speech-voice.md), a falling "Your brother." and a
# rising "Your brother?" are different renders, and ECHO wants a shorter, higher
# rise than a full question does -- so the distinction has to survive out of
# this file, not be re-derived from a question mark downstream.
QUESTION = "Q"     # a full question. Rising. Gets "?"
STATEMENT = "S"    # a complete sentence. Falling. Gets "."
ECHO = "E"         # the user's own word handed back as a prompt. Gets "?"

_WH = ("WHAT", "WHY", "WHO", "WHEN", "WHERE", "HOW", "WHICH", "WHOM", "WHOSE")

_AUX = ("DO", "DOES", "DID", "IS", "ARE", "AM", "WAS", "WERE", "CAN", "COULD",
        "WOULD", "SHOULD", "SHALL", "WILL", "HAVE", "HAS", "HAD", "SUPPOSE",
        "WHETHER", "ARENT", "DONT", "DOESNT", "ISNT", "WASNT", "CANT", "WONT")

# Enough of a verb list to tell "PLEASE GO ON" from "YOUR 4". Only consulted for
# templates short enough to be fragments, so it does not have to be complete.
_VERBS = ("SEE", "THINK", "FEEL", "KNOW", "BELIEVE", "WANT", "NEED", "TELL",
          "SAY", "SAYS", "SAID", "GO", "GOT", "COME", "MEAN", "MEANS",
          "SUGGEST", "SUGGESTS", "HELP", "LIKE", "WISH", "REMEMBER", "FORGET",
          "TRIED", "TRY", "APOLIGIZE", "APOLOGIZE", "DISCUSS", "EXPLAIN",
          "TALK", "UNDERSTAND", "SEEM", "SEEMS", "MAKES", "MADE", "MENTIONED",
          "BRING", "REMINDS", "DEPENDS", "BELONGS", "ASK", "TAKE", "LET",
          "LETS", "STATE", "ELABORATE", "PROVE", "ENJOY", "HATE", "INSIST",
          "BOTHERS", "HEAR", "MIND", "BE", "BEEN", "GETTING", "WANTING",
          "THINKING", "SURE", "CONCERNED", "WORRIED", "WORRY", "PLEASE")

_NEGATED_AUX = ("DONT", "DON'T", "CANT", "CAN'T", "DOESNT", "DOESN'T",
                "ARENT", "AREN'T", "ISNT", "ISN'T", "WONT", "WON'T",
                "WOULDNT", "WOULDN'T", "HAVENT", "HAVEN'T", "COULDNT",
                "COULDN'T", "DIDNT", "DIDN'T")

# Hand-set moods for templates the heuristic reads wrongly. Kept short on
# purpose -- if this grows past a handful the heuristic is the thing to fix.
MOOD_OVERRIDES = {
    # Elliptical, but a genuine question rather than an echo of the user.
    "SOMEONE SPECIAL PERHAPS": QUESTION,
}

# Spelling corrections to the script's own text.
#
# APOLIGIZE is Weizenbaum's typo, present in the 1966 CACM appendix at the
# SORRY keyword. It survived because a teletype transcript is a working
# document; on a 200x200 panel it is a misspelling in 16-pixel type with
# nothing else on screen, and it reads as a bug in this program rather than as
# a period detail. Same trade as I -> ME.
SPELLING = {
    "APOLIGIZE": "APOLOGIZE",
}


def respell(template):
    """Apply SPELLING. Runs before punctuate(), so words are still bare."""
    return " ".join(SPELLING.get(word, word) for word in template.split())


def mood_of(template):
    """QUESTION, STATEMENT or ECHO for one reassembly.

    ECHO is the case that prompted all this: a template that is essentially the
    user's own word handed straight back, with at most a couple of words of
    framing and no verb of its own -- "YOUR 4", "REALLY, 2", "BUT YOUR 3".
    Those are prompts, not sentences, and a full stop would be a lie about what
    they are.
    """
    if template in MOOD_OVERRIDES:
        return MOOD_OVERRIDES[template]

    # A reply of two sentences takes its mood from the last one: "HOW DO YOU
    # DO. PLEASE STATE YOUR PROBLEM" ends in an imperative, and the wh-word in
    # the greeting half must not turn the whole thing into a question.
    if "." in template[:-1]:
        return mood_of(template.rsplit(".", 1)[1].strip())

    words = template.split()
    plain = [w.strip(",.'-").upper() for w in words]

    # A negated auxiliary *opening* the reply is a question -- "DON'T YOU KNOW",
    # "CAN'T YOU BE MORE POSITIVE". Mid-sentence it is not: "I DON'T UNDERSTAND
    # THAT" and the imperative "PLEASE DON'T APOLOGIZE" are both statements.
    if plain and plain[0] in _NEGATED_AUX:
        return QUESTION

    # A tag question is a question however it started: "YOU HAVE A PARTICULAR
    # PERSON IN MIND, DON'T YOU", "YOU'RE NOT REALLY TALKING ABOUT ME - ARE YOU".
    if len(plain) >= 2 and plain[-1] in ("YOU", "I", "IT", "THEY"):
        if plain[-2] in _NEGATED_AUX or plain[-2] in _AUX:
            if "," in template or "-" in template:
                return QUESTION

    # A wh-word anywhere makes it a question: "IN WHAT WAY", "WHO, FOR EXAMPLE".
    if any(w in _WH for w in plain):
        return QUESTION
    if plain and plain[0] in _AUX:
        return QUESTION

    # A contraction elsewhere carries its own verb -- "THAT'S", "I'VE".
    if any("'" in w for w in words):
        return STATEMENT

    # The copulas count as verbs wherever they appear, not just as an opener:
    # "POSSIBLY THEY ARE 3" and "PERHAPS I WAS 4" are sentences, and a question
    # mark on either would be a different reply from the one Weizenbaum wrote.
    framing = [w for w in plain if not w.isdigit()]
    if len(framing) <= 3 and not any(w in _VERBS or w in _AUX for w in framing):
        return ECHO
    return STATEMENT


def punctuate(template, mood):
    """Give a reply the terminal mark its mood implies.

    Applied here rather than in the display code so that every consumer -- the
    panel, the REPL, and eventually the voice build -- gets the same text, and
    so a reply cannot reach a screen unpunctuated because a caller forgot.
    """
    if template[-1:] in ("?", ".", "!"):
        return template
    return template + ("." if mood == STATEMENT else "?")


def parse(text):
    """The script's parenthesised list structure -> nested Python lists."""
    tokens = re.findall(r"\(|\)|[^\s()]+", text)
    stack = [[]]
    for token in tokens:
        if token == "(":
            node = []
            stack[-1].append(node)
            stack.append(node)
        elif token == ")":
            stack.pop()
        else:
            stack[-1].append(token)
    return stack[0]


def strip_comments(text):
    return "\n".join(l for l in text.split("\n") if not l.strip().startswith(";"))


def is_rule(node):
    """A (decomposition, reassembly...) block, as opposed to a rank or a tag."""
    return isinstance(node, list) and node and isinstance(node[0], list)


def extend_class(members):
    """Apply EXTRA_CLASS_MEMBERS to one (* ...) word class."""
    out = list(members)
    for anchor, extra in EXTRA_CLASS_MEMBERS.items():
        if anchor in members:
            out.extend(w for w in extra if w not in out)
    return tuple(out)


def compile_decomposition(node):
    """Script decomposition -> tuple of ints, literals, word classes and tags.

    Element encoding, which the engine's matcher mirrors:
        0            any run of words, including none
        n > 0        exactly n words
        "WORD"       that literal word
        ("A", "B")   any one of these words   -- the script's (* A B) form
        "/TAG"       any word carrying that DLIST tag
    """
    out = []
    for item in node:
        if isinstance(item, str):
            out.append(int(item) if item.isdigit() else item)
        elif item and item[0] == "*":
            out.append(extend_class(tuple(item[1:])))
        elif item and item[0].startswith("/"):
            # (/FAMILY) and (/BELIEF) -- a reference to the DLIST tags below.
            out.append(item[0])
        else:
            # (*SAD UNHAPPY) with no space after the star tokenises this way.
            head = item[0]
            if head.startswith("*"):
                out.append(extend_class(tuple([head[1:]] + item[1:])))
            else:
                raise ValueError("unhandled decomposition element: %r" % (item,))
    return tuple(out)


def classify(template, decomposition):
    """Which kind of reply this template is, given what its slots refer to.

    The distinction that matters is whether a slot's content is recoverable
    from anything short of a real transcript:

        CANNED   no slots at all
        LITERAL  every slot refers to a component the decomposition pinned to
                 a literal word or word class -- so a keyword spotter that
                 heard that word can fill it
        NOUN     a slot refers to an arbitrary run, but sits after a word that
                 a bare noun follows grammatically
        PHRASE   a slot refers to an arbitrary run in a position that needs a
                 real clause. Unreachable without a transcript.
    """
    words = template.split()
    slots = [i for i, w in enumerate(words) if w.isdigit()]
    if not slots:
        return CANNED

    arbitrary = False
    for i in slots:
        index = int(words[i])
        if 1 <= index <= len(decomposition):
            component = decomposition[index - 1]
            # Only a bare 0 is an arbitrary run; a literal, word class, tag or
            # fixed count all pin the component to something knowable.
            if isinstance(component, int) and component == 0:
                arbitrary = True
        else:
            arbitrary = True

    if not arbitrary:
        return LITERAL
    if all(i > 0 and words[i - 1] in NOUN_CARRIERS for i in slots):
        return NOUN
    return PHRASE


def compile_template(node, decomposition):
    """One reassembly -> (kind, payload).

    Most are (kind, "TEXT WITH 3 SLOTS"). Three script constructs are control
    rather than reply, and carry a different payload:

        ("G", "WHAT")             (=WHAT)  -- answer as if WHAT had matched
        ("K", None)               (NEWKEY) -- give up, try the next keyword
        ("P", ("I ARE 3", "YOU")) (PRE (I ARE 3) (=YOU)) -- rewrite, then goto
    """
    if node and node[0] == "NEWKEY":
        return (NEWKEY, None, None)

    if node and node[0] == "PRE":
        rewrite = " ".join(str(t) for t in node[1])
        target = node[2][0].lstrip("=")
        return (PRE, None, (rewrite, target))

    text = " ".join(t if isinstance(t, str) else " ".join(t) for t in node)
    if text.startswith("="):
        return (GOTO, None, text[1:])
    text = respell(text)
    # PRE rewrites are input, not output, so they are never punctuated; every
    # other reply is, here, once, for every consumer.
    mood = mood_of(text)
    return (classify(text, decomposition), mood, punctuate(text, mood))


def build(top):
    subs = {}          # word -> replacement, applied to the whole input
    tags = {}          # word -> DLIST tag, for the (/FAMILY) decompositions
    keywords = {}      # word -> (rank, goto, rules)
    memory = None
    none = None
    greeting = None

    for form in top:
        if isinstance(form, str):
            continue  # the bare START marker
        if not form:
            continue

        head = form[0]

        # The opening line is a bare sentence, not a keyword entry.
        if greeting is None and not is_rule(form) and " ".join(form).startswith("HOW DO YOU DO"):
            greeting = punctuate(respell(" ".join(form)),
                                 mood_of(" ".join(form)))
            continue

        if head == "MEMORY":
            # (MEMORY MY (0 YOUR 0 = LETS DISCUSS FURTHER WHY YOUR 3) ...)
            templates = []
            for block in form[2:]:
                split = block.index("=")
                text = respell(" ".join(block[split + 1:]))
                templates.append((
                    compile_decomposition(block[:split]),
                    mood_of(text),
                    punctuate(text, mood_of(text)),
                ))
            memory = (form[1], tuple(templates))
            continue

        rest = form[1:]
        rank = 0
        goto = None

        # (DONT = DON'T) and (I = YOU ...rules...). The script folds ELIZA's
        # person-swap into this table rather than reflecting captured text
        # afterwards, so the decompositions below are written against already
        # swapped input -- "I WANT" is matched as "YOU WANT". Getting this
        # backwards is the classic way to make ELIZA answer in the wrong person.
        if len(rest) >= 2 and rest[0] == "=":
            subs[head] = rest[1]
            rest = rest[2:]

        if rest and isinstance(rest[0], str) and rest[0].isdigit():
            rank = int(rest[0])
            rest = rest[1:]

        for index, item in enumerate(rest):
            # (MOTHER DLIST(/NOUN FAMILY)) tokenises as the bare word DLIST
            # followed by the tag list, not as one nested form.
            if item == "DLIST" and index + 1 < len(rest):
                for tag in rest[index + 1]:
                    tag = tag.lstrip("/")
                    if tag:
                        tags.setdefault(head, []).append(tag)
                continue
            if not isinstance(item, list):
                continue
            # (HOW (=WHAT)) -- a keyword that owns no rules of its own.
            if len(item) == 1 and isinstance(item[0], str) and item[0].startswith("="):
                goto = item[0][1:]

        rules = []
        for block in rest:
            if not is_rule(block):
                continue
            decomposition = compile_decomposition(block[0])
            templates = tuple(compile_template(t, decomposition) for t in block[1:])
            rules.append((decomposition, templates))

        if head == "NONE":
            none = tuple(t[2] for t in rules[0][1])
            continue

        if rules or goto:
            keywords[head] = (rank, goto, tuple(rules))

    # DLIST entries that only ever appear as a substitution ((MOM = MOTHER
    # DLIST(/ FAMILY))) inherit their target's tag, which is what makes "MOM"
    # match a (/FAMILY) decomposition after substitution.
    for word, target in subs.items():
        if target in tags and word not in tags:
            tags[word] = list(tags[target])

    for word, extra in EXTRA_TAGS.items():
        tags.setdefault(word, []).extend(t for t in extra if t not in tags.get(word, []))

    return greeting, subs, {k: tuple(v) for k, v in tags.items()}, keywords, memory, none


def render(greeting, subs, tags, keywords, memory, none):
    out = []
    w = out.append

    w('"""Weizenbaum\'s 1966 DOCTOR script as Python data. Generated -- do not edit.')
    w("")
    w("Regenerate with `uv run tools/make_eliza_rules.py`. The interesting")
    w("decisions (which template is fillable from what) live in that script, not")
    w("here; edit it and re-run rather than patching this file.")
    w("")
    w("Reply kinds tag each template with what it needs in order to be spoken:")
    w("")
    w("    CANNED   no slots -- always usable")
    w("    LITERAL  slots refer to a word the decomposition pinned down, so a")
    w("             keyword spotter that heard that word can fill them")
    w("    NOUN     slots refer to the user's own words, but sit where a bare")
    w("             noun is grammatical, so a spotted noun can stand in")
    w("    PHRASE   slots need a real clause. Reachable only from a transcript.")
    w("")
    w("A bag-of-keywords front end can use CANNED, LITERAL and NOUN. Only a full")
    w("transcript unlocks PHRASE. That split is the whole reason this is data.")
    w("")
    w("Control kinds carry no reply text of their own:")
    w("")
    w('    GOTO     ("G", "WHAT")             answer as if WHAT had matched')
    w('    NEWKEY   ("K", None)               abandon this keyword, try the next')
    w('    PRE      ("P", ("I ARE 3", "YOU")) rewrite the input, then goto')
    w("")
    w("Every reply also carries a mood, and the terminal punctuation that goes")
    w("with it. The 1966 script has none: on a teletype the next prompt line")
    w("delimited a reply, but alone on a 200x200 panel the same text reads as")
    w("unfinished, and the short ones read as broken.")
    w("")
    w("    QUESTION   a full question. Rising. Gets \"?\"")
    w("    STATEMENT  a complete sentence. Falling. Gets \".\"")
    w("    ECHO       the user's own word handed back as a prompt. Gets \"?\"")
    w("")
    w("ECHO is separate from QUESTION because it is the case that reads as a")
    w("fragment -- \"YOUR 4\", \"REALLY, 2\" -- and because spoken it wants a")
    w("shorter, higher rise than a full question. Control forms carry None.")
    w('"""')
    w("")
    w('CANNED = "C"')
    w('LITERAL = "L"')
    w('NOUN = "N"')
    w('PHRASE = "T"')
    w("")
    w('QUESTION = "Q"')
    w('STATEMENT = "S"')
    w('ECHO = "E"')
    w("")
    w('GOTO = "G"')
    w('NEWKEY = "K"')
    w('PRE = "P"')
    w("")
    w("# Kinds a reply can actually be built from without a transcript.")
    w("SPOTTABLE = (CANNED, LITERAL, NOUN)")
    w("")
    w("GREETING = %r" % greeting)
    w("")
    w("# Applied to every input word before anything else. This is also ELIZA's")
    w("# person swap -- I/YOU, MY/YOUR -- which is why the decompositions read as")
    w("# though the user had said them about the doctor.")
    w("SUBS = {")
    for key in sorted(subs):
        w("    %r: %r," % (key, subs[key]))
    w("}")
    w("")
    w("# The script's DLIST tags, referenced from decompositions as \"/FAMILY\".")
    w("TAGS = {")
    for key in sorted(tags):
        w("    %r: %r," % (key, tags[key]))
    w("}")
    w("")
    w("# keyword -> (rank, goto, ((decomposition, ((kind, mood, payload), ...)), ...))")
    w("#")
    w("# Rank arbitrates when the input contains several keywords; the script")
    w("# gives COMPUTER 50 so that talking about machines wins over anything else.")
    w("RULES = {")
    for key in sorted(keywords):
        rank, goto, rules = keywords[key]
        w("    %r: (%d, %r, (" % (key, rank, goto))
        for decomposition, templates in rules:
            w("        (%r, (" % (decomposition,))
            for kind, mood, payload in templates:
                w("            (%r, %r, %r)," % (kind, mood, payload))
            w("        )),")
        w("    )),")
    w("}")
    w("")
    w("# The MEMORY queue: when the input contains no keyword at all, ELIZA may")
    w("# answer something the user said several turns ago instead of admitting it")
    w("# understood nothing. It is the single cheapest trick in the script.")
    w("MEMORY_KEYWORD = %r" % memory[0])
    w("MEMORY = (")
    for decomposition, mood, template in memory[1]:
        w("    (%r, %r, %r)," % (decomposition, mood, template))
    w(")")
    w("")
    w("# Last resort, when there is no keyword and nothing in memory.")
    w("NONE = (")
    for template in none:
        w("    %r," % template)
    w(")")
    w("")
    return "\n".join(out)


def main():
    if len(sys.argv) > 1:
        text = open(sys.argv[1]).read()
    else:
        print("fetching %s" % SCRIPT_URL)
        text = urllib.request.urlopen(SCRIPT_URL).read().decode("utf-8")

    top = parse(strip_comments(text))
    greeting, subs, tags, keywords, memory, none = build(top)
    source = render(greeting, subs, tags, keywords, memory, none)

    with open(OUT_PATH, "w") as handle:
        handle.write(source)

    counts = {}
    for rank, goto, rules in keywords.values():
        for decomposition, templates in rules:
            for kind, _mood, payload in templates:
                counts[kind] = counts.get(kind, 0) + 1
    total = sum(counts.get(k, 0) for k in (CANNED, LITERAL, NOUN, PHRASE))

    print("%s: %d keywords, %d decompositions, %d templates"
          % (OUT_PATH, len(keywords),
             sum(len(r[2]) for r in keywords.values()), total))
    for kind, label in ((CANNED, "canned"), (LITERAL, "literal-slot"),
                        (NOUN, "noun-slot"), (PHRASE, "needs a phrase")):
        n = counts.get(kind, 0)
        print("  %-16s %3d  (%2.0f%%)" % (label, n, 100.0 * n / total))
    usable = sum(counts.get(k, 0) for k in (CANNED, LITERAL, NOUN))
    print("  usable from a bag of keywords: %d of %d (%.0f%%)"
          % (usable, total, 100.0 * usable / total))
    for kind, label in ((GOTO, "goto"), (NEWKEY, "newkey"), (PRE, "pre")):
        print("  %-16s %3d" % (label, counts.get(kind, 0)))
    print("  %d substitutions, %d tagged words, %d NONE fallbacks"
          % (len(subs), len(tags), len(none)))


if __name__ == "__main__":
    sys.exit(main())
