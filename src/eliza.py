"""ELIZA's DOCTOR script, run against either a sentence or a bag of keywords.

Deliberately does no I/O at all -- no display, no audio, no `machine` -- so the
whole thing can be exercised on the host with `tools/eliza_repl.py` and
`tools/test_eliza.py`. The rule data lives in `eliza_rules`, generated from
Weizenbaum's 1966 script by `tools/make_eliza_rules.py`.

## The input boundary is the design

There are two entry points, and the difference between them is the entire
question this module exists to answer:

    respond_to_words(["I", "HATE", "MY", "MOTHER"])   full ELIZA
    respond_to_keywords(["MY", "MOTHER"])             degraded ELIZA

The first has word order, so it can run the script's decomposition patterns and
capture arbitrary runs of the user's own words -- which is what lets it say
"WHY DO YOU HATE YOUR MOTHER". The second is what a small-vocabulary keyword
spotter can actually deliver: a set of recognised words, no order, and no idea
what else was said.

Of the 191 reply templates in the script, 100 survive the second path and 91 do
not, because they need a clause the spotter never heard. `eliza_rules` tags
every template with which case it is, so the filtering is data rather than a
pile of conditionals here. See that module's header.

The degraded path is not a fallback for when matching fails; it is a different
front end for the same rules, and it is the one the device is likely to use.

## Rotation, not randomness

DOCTOR cycles through a rule's replies in order and wraps around. That is not an
implementation detail to modernise away -- it is what stops the toy repeating
itself within a conversation, which random choice does constantly. So there is
no RNG on the main path. `_rand_below` is used in exactly one place: choosing
between several equally good spotted nouns, where the script has no opinion
because it never anticipated this input.
"""

import os

import eliza_rules as rules

# The script breaks input at these before matching, and keeps the first fragment
# that contains a keyword. "I am sad, but I am fine" is about being sad.
_BREAKS = (",", ".", "!", "?", ";")
_BREAK_WORD = "BUT"

# Function words a *bag* match may assume it heard. These are the glue the
# script's patterns are built from -- "(0 YOUR 0 /FAMILY 0)" is really about the
# family word, and "(0 YOU ARE 0 (*SAD UNHAPPY DEPRESSED SICK) 0)" is really
# about SAD -- and they are exactly the words nobody would spend a 25-word
# spotter vocabulary on.
#
# Without this the degraded path cannot reach the script's best rules at all.
# The family responses hang off keyword MY, the feeling ones off keyword I, and
# a spotter hears neither; every noun utterance falls through to a NONE
# deflection. Someone who says "mother" to a therapist means "my mother"
# overwhelmingly often, and supplying that possessive is the same kind of
# judgement the script makes freely elsewhere.
#
# Assuming is licensed only when a content word was actually heard -- see
# _heard_content. Without that guard an empty bag would start matching rules,
# which is how a toy begins answering things nobody said.
_ASSUMED_IN_BAG = ("ARE", "AM", "IS", "WAS", "I", "YOU", "MY", "YOUR", "ME")

# Stripped outright; unlike _BREAKS these carry no clause boundary.
_PUNCTUATION = ":\"()"


def _rand_below(n):
    """Uniform random int in [0, n), via the RP2350 hardware RNG.

    Duplicated from magic8 rather than imported, following the precedent set by
    shake.py: it keeps this module independent of the Magic 8-Ball, and it is
    six lines. Rejection sampling because plain modulo would bias the low end.
    """
    limit = 256 - (256 % n)
    while True:
        value = os.urandom(1)[0]
        if value < limit:
            return value % n


def normalise(text):
    """A sentence -> the upper-case word list the rules are written against.

    A trailing comma or full stop is left attached rather than stripped,
    because the script breaks the input at those and answers only the first
    clause -- "I need help, that much seems certain" is a request for help.
    _fragment removes them once it has used them.
    """
    out = []
    for word in text.split():
        word = word.strip(_PUNCTUATION).upper()
        # Strip break characters only from the front; a trailing one is data.
        word = word.lstrip("".join(_BREAKS))
        if word:
            out.append(word)
    return out


def sentence_case(text):
    """Script text -> something worth putting on a 200x200 panel.

    The script is all upper case because it was written for a teletype. Left as
    presentation rather than baked into the data, so the rules stay a faithful
    transcription and the caller decides.
    """
    if not text:
        return text
    words = []
    # Several replies are two sentences ("HOW DO YOU DO. PLEASE STATE YOUR
    # PROBLEM"), so the capital has to follow the full stop, not just lead.
    start = True
    for word in text.lower().split():
        if start or word == "i" or word.startswith("i'"):
            word = word[0].upper() + word[1:]
        start = word[-1:] in (".", "!", "?")
        words.append(word)
    return " ".join(words)


# --- matching ---------------------------------------------------------------
# The script's decomposition patterns are word lists, not regexes, and they are
# reproduced as word lists here. MicroPython's `re` has neither `\b` nor
# counted repetitions, which are precisely the two things these patterns need,
# so translating them to regex would be both lossy and slower.


def _element_matches(element, word, tags):
    if isinstance(element, tuple):     # (* SAD UNHAPPY DEPRESSED SICK)
        return word in element
    if element[0] == "/":              # (/FAMILY)
        return element[1:] in tags.get(word, ())
    return element == word


def match(pattern, words, tags=None):
    """Decompose `words` by `pattern`, or None.

    Returns one captured word list per pattern element, so a reassembly's "3"
    can index straight into the result. Element encoding is documented in
    eliza_rules.
    """
    if tags is None:
        tags = rules.TAGS
    return _match(pattern, 0, words, 0, [], tags)


def _match(pattern, pi, words, wi, captured, tags):
    if pi == len(pattern):
        return captured if wi == len(words) else None

    element = pattern[pi]

    if isinstance(element, int):
        if element:
            # Exactly n words.
            if wi + element <= len(words):
                return _match(pattern, pi + 1, words, wi + element,
                              captured + [words[wi:wi + element]], tags)
            return None
        # Any run, shortest first -- the script's patterns assume the wildcard
        # gives up as little as it can, so that "(0 YOU 0 ME)" binds the *first*
        # YOU rather than the last.
        for take in range(0, len(words) - wi + 1):
            found = _match(pattern, pi + 1, words, wi + take,
                           captured + [words[wi:wi + take]], tags)
            if found is not None:
                return found
        return None

    if wi < len(words) and _element_matches(element, words[wi], tags):
        return _match(pattern, pi + 1, words, wi + 1,
                      captured + [[words[wi]]], tags)
    return None


def fill(template, captured):
    """Substitute a reassembly's digit slots with the captured components.

    One deliberate departure from the script. Its substitution table maps
    YOU -> I in one pass with no notion of grammatical case, so "what I told
    you" comes back as "what you told I". That is authentic -- the 1966 program
    does it too, and it is one of the seams people noticed -- but on a 200x200
    panel with one sentence on it, it reads as a bug rather than as a period
    detail. A trailing "I" inside a captured phrase is therefore rendered "ME".

    Deliberately narrow: only at the end of a multi-word capture, which is the
    only place the case can be wrong. A capture that is the bare word "I" is
    left alone, because there it is the subject and correct.
    """
    out = []
    for word in template.split():
        if word.isdigit():
            index = int(word)
            if 1 <= index <= len(captured):
                component = captured[index - 1]
                if len(component) > 1 and component[-1] == "I":
                    component = component[:-1] + ["ME"]
                piece = " ".join(component)
                if piece:
                    out.append(piece)
            continue
        out.append(word)
    return " ".join(out)


class Doctor:
    """One conversation. Holds the rotation counters and the memory queue."""

    # How many turns of unrecognised input before a queued memory is offered.
    # The script's own trigger is simply "no keyword at all"; the delay is
    # ours, because firing it on the very first miss gives the game away.
    MEMORY_AFTER_MISSES = 1

    def __init__(self, ruleset=None, priority=None):
        # Injectable so a test can drive a two-keyword script, and so a device
        # build can ship a filtered subset without this module knowing.
        self.rules = ruleset or rules

        # Extra rank per word, added to the script's own. Needed only on the
        # degraded path, and needed badly there: the script leaves 27 of its 45
        # keywords at rank 0 because it expects word order to break the tie, and
        # a bag has no order. Left to alphabetical chance, "I AM UNHAPPY" is
        # answered by AM -- which just defers to WHAT and says "WHY DO YOU ASK"
        # -- instead of by the rule that would have said "I AM SORRY TO HEAR YOU
        # ARE UNHAPPY". Ranking the vocabulary is how a spotter build gets its
        # own say in which words matter, and it is free.
        self.priority = priority or {}

        self._turn = {}       # (keyword, decomposition index) -> replies used
        self._memory = []     # queued replies about things said earlier
        self._misses = 0
        self._classes = None  # word_classes(), computed once on demand

    def greet(self):
        return self.rules.GREETING

    # --- input side: two front ends over one rule set -----------------------

    def respond(self, text):
        """Convenience wrapper: a typed sentence."""
        return self.respond_to_words(normalise(text))

    def respond_to_words(self, words):
        """Full ELIZA. `words` is an ordered, upper-case word list.

        Only this path can produce the PHRASE templates -- the ones that quote a
        run of the user's own sentence back at them.
        """
        words = self._fragment(words)
        keywords = self._rank(words)
        swapped = [self.rules.SUBS.get(w, w) for w in words]

        self._remember(words, swapped)

        reply = self._answer(keywords, swapped, spotted=None)
        if reply is not None:
            self._misses = 0
            return reply
        return self._give_up()

    def respond_to_keywords(self, spotted, nouns=None):
        """Degraded ELIZA, for a keyword spotter's output.

        `spotted` is an unordered collection of recognised words; anything the
        user said that the spotter does not know simply is not here, and there
        is no way to tell a two-word utterance from a thirty-word one.

        `nouns` is the subset of the vocabulary worth echoing back. It defaults
        to the words the script itself tags (its /FAMILY and /NOUN lists), but a
        device build should pass its own -- the whole argument for spending the
        spotter's vocabulary on nouns is that these are what make the toy sound
        like it heard you.
        """
        spotted = [w.upper() for w in spotted]
        if nouns is None:
            nouns = self.default_nouns()
        swapped = [self.rules.SUBS.get(w, w) for w in spotted]

        available = [self.rules.SUBS.get(w, w) for w in spotted if w in nouns]

        # Keywords actually heard come first. Then, if a content word was heard,
        # the keywords whose rules would fire once the missing function words
        # are supplied -- which is the only way the degraded path ever reaches
        # the family and feeling rules, since those hang off MY and I.
        keywords = self._rank(spotted)
        if self._heard_content(spotted, nouns):
            for keyword in self._assumable(swapped, available):
                if keyword not in keywords:
                    keywords.append(keyword)

        reply = self._answer(keywords, swapped, spotted=available)
        if reply is not None:
            self._misses = 0
            return reply
        return self._give_up()

    def default_nouns(self):
        """Words the script itself considers echo-worthy nouns."""
        out = []
        for word, tags in self.rules.TAGS.items():
            if "FAMILY" in tags or "NOUN" in tags:
                out.append(word)
        return out

    def word_classes(self):
        """Every word the script names in a (* A B C) alternation.

        These are the script's own closed vocabularies -- SAD/UNHAPPY/
        DEPRESSED/SICK and so on -- which is to say the words it was already
        prepared to spot rather than parse. Cached: the walk is cheap but it is
        the same answer every turn.
        """
        if self._classes is None:
            found = []
            for rank, goto, ruleset in self.rules.RULES.values():
                for pattern, templates in ruleset:
                    for element in pattern:
                        if isinstance(element, tuple):
                            for word in element:
                                if word not in found:
                                    found.append(word)
            self._classes = found
        return self._classes

    def _heard_content(self, spotted, nouns):
        """Did the bag contain a word with something to say?

        This is what licenses assuming the function words. A noun to echo or a
        word the script has a class for is real evidence about the subject;
        anything else and we would be inventing a sentence rather than
        completing one.
        """
        classes = self.word_classes()
        tags = self.rules.TAGS
        for word in spotted:
            if word in nouns or word in classes or word in tags:
                return True
        return False

    def _binds_a_heard_word(self, pattern, words):
        """Does this decomposition pin anything we actually heard?

        The guard that keeps assumption from becoming invention. Without it,
        every rule whose decomposition is a bare "(0)" matches any bag at all --
        and since COMPUTER is rank 50, the degraded path answers "DO COMPUTERS
        WORRY YOU" to somebody talking about their mother.
        """
        for element in pattern:
            if isinstance(element, int):
                continue
            if isinstance(element, tuple):
                if any(w in element for w in words):
                    return True
            elif element[0] == "/":
                tag = element[1:]
                if any(tag in self.rules.TAGS.get(w, ()) for w in words):
                    return True
            elif element in words:
                return True
        return False

    def _assumable(self, words, available):
        """Keywords whose rules could fire if the function words were supplied.

        Reuses the ordinary bag matcher as the reachability test rather than
        hard-coding "MY": a rule is reachable when every pinned element of one
        of its decompositions is either something we heard or something in
        _ASSUMED_IN_BAG, and at least one template survives the kind filter.

        A reachable rule still has to be *about* something. It qualifies either
        because it pins a word we heard -- "(0 YOUR 0 /FAMILY 0)" pins MOTHER --
        or because it can echo a spotted noun into a slot, which is how a noun
        the script never tagged, like WORK, still gets repeated back. Ordered by
        rank so the strongest reachable rule is tried first.
        """
        found = []
        for keyword, (rank, goto, ruleset) in self.rules.RULES.items():
            if goto:
                continue
            for pattern, templates in ruleset:
                captured = self._captured_from_spotted(pattern, words, available)
                if captured is None:
                    continue
                usable = self._usable(templates, available, captured)
                if not usable:
                    continue
                echoes = available and any(k == self.rules.NOUN for k, _ in usable)
                if not (self._binds_a_heard_word(pattern, words) or echoes):
                    continue
                found.append((-(rank + self.priority.get(keyword, 0)), keyword))
                break
        found.sort()
        return [keyword for _, keyword in found]

    # --- the shared middle --------------------------------------------------

    def _fragment(self, words):
        """First clause containing a keyword, as the script does."""
        fragments = [[]]
        for word in words:
            if word == _BREAK_WORD:
                fragments.append([])
                continue
            broke = word[-1:] in _BREAKS
            word = word.rstrip("".join(_BREAKS))
            if word:
                fragments[-1].append(word)
            if broke:
                fragments.append([])
        fragments = [f for f in fragments if f]
        for fragment in fragments:
            if any(w in self.rules.RULES for w in fragment):
                return fragment
        return fragments[0] if fragments else []

    def _rank(self, words):
        """Keywords present, best first.

        Rank is the script's own arbitration: COMPUTER is 50 so that a mention
        of machines beats everything else in the sentence. Ties keep the order
        the user said them in, which is also what the original does.
        """
        found = []
        for position, word in enumerate(words):
            entry = self.rules.RULES.get(word)
            if entry is not None and word not in [f[2] for f in found]:
                rank = entry[0] + self.priority.get(word, 0)
                found.append((-rank, position, word))
        found.sort()
        return [word for _, _, word in found]

    def _answer(self, keywords, words, spotted):
        """Walk the ranked keywords until one produces a reply.

        `spotted` is None on the ordered path and a list of echo-able nouns on
        the degraded one; it is what tells the two apart from here down.
        """
        pending = list(keywords)
        seen = set()

        while pending:
            keyword = pending.pop(0)
            if keyword in seen:
                continue
            seen.add(keyword)

            entry = self.rules.RULES.get(keyword)
            if entry is None:
                continue
            rank, goto, ruleset = entry
            if goto:
                pending.insert(0, goto)
                continue

            reply, control = self._try_keyword(keyword, ruleset, words, spotted)
            if reply is not None:
                return reply
            if control is not None:
                # A goto or a PRE rewrite. Re-enter with the new keyword, and
                # in the PRE case with rewritten words.
                target, rewritten = control
                if rewritten is not None:
                    words = rewritten
                pending.insert(0, target)
        return None

    def _try_keyword(self, keyword, ruleset, words, spotted):
        for index, (pattern, templates) in enumerate(ruleset):
            if spotted is None:
                captured = match(pattern, words, self.rules.TAGS)
                if captured is None:
                    continue
            else:
                # The degraded path cannot verify the pattern -- it has no word
                # order and no unrecognised words. It uses the decomposition
                # only as a key to what each slot *means*, then fills the slots
                # it can from what was spotted. This is the honest degradation:
                # we are no longer checking that the user said it that way.
                captured = self._captured_from_spotted(pattern, words, spotted)
                if captured is None:
                    continue

            usable = self._usable(templates, spotted, captured)
            if not usable:
                continue

            key = (keyword, index)
            position = self._turn.get(key, 0)
            self._turn[key] = position + 1
            kind, payload = usable[position % len(usable)]

            if kind == self.rules.NEWKEY:
                continue                       # this keyword declines to answer
            if kind == self.rules.GOTO:
                return None, (payload, None)
            if kind == self.rules.PRE:
                rewrite, target = payload
                return None, (target, fill(rewrite, captured).split())
            # A NOUN template was already filled by _usable, which had to plant
            # the noun before it could tell whether the template was usable at
            # all. Filling again is a no-op on it, and the single path is worth
            # more than saving the call.
            return fill(payload, captured), None
        return None, None

    def _captured_from_spotted(self, pattern, words, spotted):
        """Fake a decomposition from a bag of words.

        Produces one component per pattern element, filled where the bag can
        say something and left empty where it cannot. Returns None if the
        pattern's pinned elements were not spotted at all, which is the only
        check still available -- we can tell that MOTHER was said, just not
        where in the sentence.
        """
        captured = []
        for element in pattern:
            if isinstance(element, int):
                # An arbitrary run. A spotted noun may stand in for it later,
                # for the templates tagged NOUN; nothing else can.
                captured.append([])
            elif isinstance(element, tuple):
                hit = [w for w in words if w in element]
                if not hit:
                    return None
                captured.append([hit[0]])
            elif element[0] == "/":
                tag = element[1:]
                hit = [w for w in words if tag in self.rules.TAGS.get(w, ())]
                if not hit:
                    return None
                captured.append([hit[0]])
            else:
                if element not in words and element not in _ASSUMED_IN_BAG:
                    return None
                captured.append([element])
        return captured

    def _usable(self, templates, spotted, captured):
        """Templates this input can actually fill, in script order."""
        out = []
        for kind, payload in templates:
            if kind in (self.rules.GOTO, self.rules.NEWKEY, self.rules.PRE):
                out.append((kind, payload))
                continue
            if spotted is None:
                out.append((kind, payload))
                continue
            if kind not in self.rules.SPOTTABLE:
                continue
            if kind == self.rules.NOUN:
                # The one place a noun can be dropped into a slot meant for a
                # clause. Skip the template entirely if nothing was spotted --
                # "WHY DO YOU SAY YOUR" is worse than saying something else.
                if not spotted:
                    continue
                noun = spotted[_rand_below(len(spotted))]
                out.append((kind, self._plant(payload, captured, noun)))
                continue
            out.append((kind, payload))
        return out

    def _plant(self, template, captured, noun):
        """Put `noun` into every empty slot of a NOUN template."""
        filled = list(captured)
        for word in template.split():
            if word.isdigit():
                index = int(word)
                if 1 <= index <= len(filled) and not filled[index - 1]:
                    filled[index - 1] = [noun]
        return fill(template, filled)

    # --- memory and giving up -----------------------------------------------

    def _remember(self, words, swapped):
        """Queue something to bring up later, as the script's MEMORY does.

        Only the ordered path can do this: it needs to capture a run of the
        user's own words, which is exactly what the degraded path lacks.

        The trigger keyword is looked up in what the user said, but the pattern
        matches what substitution left behind -- MEMORY's keyword is MY and its
        decompositions are written as "(0 YOUR 0)", the same before/after split
        that keyword lookup uses.
        """
        if self.rules.MEMORY_KEYWORD not in words:
            return
        for pattern, template in self.rules.MEMORY:
            captured = match(pattern, swapped, self.rules.TAGS)
            if captured is not None:
                self._memory.append(fill(template, captured))
                # The script keeps four; more than that and it starts dredging
                # up things far enough back to be strange rather than uncanny.
                if len(self._memory) > len(self.rules.MEMORY):
                    self._memory.pop(0)
                return

    def _give_up(self):
        """Nothing matched. Offer a memory if one is queued, else a NONE line."""
        self._misses += 1
        if self._memory and self._misses > self.MEMORY_AFTER_MISSES:
            self._misses = 0
            return self._memory.pop(0)
        key = ("NONE", 0)
        position = self._turn.get(key, 0)
        self._turn[key] = position + 1
        return self.rules.NONE[position % len(self.rules.NONE)]
