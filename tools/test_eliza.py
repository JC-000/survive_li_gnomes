#!/usr/bin/env python3
"""Tests for src/eliza.py and the generated src/eliza_rules.py.

    uv run tools/test_eliza.py
    uv run tools/test_eliza.py -v

Host-only and hardware-free by design -- `eliza` imports nothing but `os` and
its own rule data, which is what makes the engine testable at all. Everything
downstream of it (panel, codec) is not exercised here.

The tests worth reading before changing anything are the bag-mode ones. The
engine has two front ends over one rule set, and the invariant that matters is
that the degraded one can never emit a reply it cannot fill -- a stray "3" on
the panel is the visible form of that bug.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import eliza  # noqa: E402
import eliza_rules as rules  # noqa: E402


class TestMatcher(unittest.TestCase):
    """The word-list matcher, against patterns taken from the real script."""

    def decompose(self, pattern, sentence):
        found = eliza.match(pattern, sentence.split())
        return None if found is None else [" ".join(c) for c in found]

    def test_wildcard_and_literals(self):
        self.assertEqual(
            self.decompose((0, "YOU", 0, "ME"), "WHY DO YOU HATE ME"),
            ["WHY DO", "YOU", "HATE", "ME"])

    def test_leading_wildcard_binds_shortest(self):
        # "(0 YOU 0 ME)" must bind the first YOU, not the last, or the captured
        # middle swallows the wrong half of the sentence.
        self.assertEqual(
            self.decompose((0, "YOU", 0), "YOU SAID YOU WOULD"),
            ["", "YOU", "SAID YOU WOULD"])

    def test_trailing_wildcard_may_be_empty(self):
        self.assertEqual(self.decompose((0, "YOU", 0), "YOU"), ["", "YOU", ""])

    def test_no_match_returns_none(self):
        self.assertIsNone(self.decompose((0, "YOU", 0, "ME"), "WHY DO YOU HATE HIM"))

    def test_exact_word_count(self):
        self.assertEqual(self.decompose((0, "I", 1, "YOU"), "SO I HATE YOU"),
                         ["SO", "I", "HATE", "YOU"])
        self.assertIsNone(self.decompose((0, "I", 1, "YOU"), "SO I REALLY HATE YOU"))

    def test_word_class(self):
        # The script's (*SAD UNHAPPY DEPRESSED SICK) form.
        pattern = (0, "YOU", "ARE", 0, ("SAD", "UNHAPPY"), 0)
        self.assertEqual(
            self.decompose(pattern, "YOU ARE VERY SAD TODAY"),
            ["", "YOU", "ARE", "VERY", "SAD", "TODAY"])
        self.assertIsNone(self.decompose(pattern, "YOU ARE VERY TALL TODAY"))

    def test_dlist_tag(self):
        # "/FAMILY" matches any word the script tags, which is what lets one
        # rule cover mother, father, sister, brother, wife and children.
        pattern = (0, "YOUR", 0, "/FAMILY", 0)
        self.assertIsNotNone(self.decompose(pattern, "YOUR POOR SISTER AGAIN"))
        self.assertIsNone(self.decompose(pattern, "YOUR POOR DOG AGAIN"))

    def test_real_script_patterns(self):
        for pattern, sentence, expected in (
            ((0, "YOU", "REMEMBER", 0), "DO YOU REMEMBER THE WAR",
             ["DO", "YOU", "REMEMBER", "THE WAR"]),
            ((0, "YOU", "ARE", 0), "I THINK YOU ARE RIGHT",
             ["I THINK", "YOU", "ARE", "RIGHT"]),
            ((0, "YOUR", 0), "THIS IS YOUR PROBLEM",
             ["THIS IS", "YOUR", "PROBLEM"]),
        ):
            self.assertEqual(self.decompose(pattern, sentence), expected, pattern)


class TestFill(unittest.TestCase):
    def test_slots_are_one_based(self):
        captured = [["WHY", "DO"], ["YOU"], ["HATE"], ["ME"]]
        self.assertEqual(eliza.fill("WHAT MAKES YOU 3 ME", captured),
                         "WHAT MAKES YOU HATE ME")

    def test_empty_component_leaves_no_double_space(self):
        self.assertEqual(eliza.fill("REALLY, 1 THEN", [[]]), "REALLY, THEN")

    def test_out_of_range_slot_is_dropped(self):
        self.assertEqual(eliza.fill("SO 9 THEN", [["A"]]), "SO THEN")

    def test_trailing_subject_pronoun_becomes_an_object_one(self):
        # The script's one-pass YOU -> I substitution has no notion of case, so
        # "what I told you" decomposes to "what you told I".
        self.assertEqual(eliza.fill("FORGET 1", [["WHAT", "YOU", "TOLD", "I"]]),
                         "FORGET WHAT YOU TOLD ME")

    def test_a_bare_i_capture_is_left_alone(self):
        # There it really is the subject.
        self.assertEqual(eliza.fill("SO 1 THEN", [["I"]]), "SO I THEN")


class TestNormalise(unittest.TestCase):
    def test_upper_cases_and_splits(self):
        self.assertEqual(eliza.normalise("I am sad"), ["I", "AM", "SAD"])

    def test_keeps_clause_breaks_for_the_fragmenter(self):
        # The comma has to survive normalise() or the script cannot break on it.
        self.assertIn("HELP,", eliza.normalise("I need help, that much is certain"))

    def test_answers_the_first_clause_only(self):
        doctor = eliza.Doctor()
        self.assertEqual(doctor.respond("I need some help, that much seems certain"),
                         "WHAT WOULD IT MEAN TO YOU IF YOU GOT SOME HELP?")

    def test_breaks_on_but(self):
        doctor = eliza.Doctor()
        reply = doctor.respond("I am sad but the weather is fine")
        self.assertIn("SAD", reply)


class TestRanking(unittest.TestCase):
    def test_script_rank_wins(self):
        # COMPUTER is rank 50 in the script precisely so it beats everything.
        doctor = eliza.Doctor()
        self.assertEqual(doctor._rank(["MY", "COMPUTER", "YES"])[0], "COMPUTER")

    def test_ties_keep_the_order_they_were_said(self):
        doctor = eliza.Doctor()
        self.assertEqual(doctor._rank(["YES", "NO"]), ["YES", "NO"])
        self.assertEqual(doctor._rank(["NO", "YES"]), ["NO", "YES"])

    def test_priority_table_overrides_a_tie(self):
        # The whole point of the priority table: with no word order, "AM" would
        # otherwise beat "I" by sorting first, and answer with a deferral.
        plain = eliza.Doctor()
        self.assertEqual(plain._rank(["AM", "I"])[0], "AM")
        ranked = eliza.Doctor(priority={"I": 6, "AM": -4})
        self.assertEqual(ranked._rank(["AM", "I"])[0], "I")

    def test_unknown_words_are_not_keywords(self):
        self.assertEqual(eliza.Doctor()._rank(["AARDVARK", "BANANA"]), [])


class TestRotation(unittest.TestCase):
    def test_replies_cycle_rather_than_repeat(self):
        # The script cycles its replies; random choice would repeat constantly,
        # which is what makes a toy feel broken.
        doctor = eliza.Doctor()
        replies = [doctor.respond("Everyone hates me") for _ in range(4)]
        self.assertEqual(len(set(replies)), 4, replies)

    def test_cycle_wraps_around(self):
        doctor = eliza.Doctor()
        first = doctor.respond("Everyone hates me")
        count = len(rules.RULES["EVERYONE"][2][0][1])
        for _ in range(count - 1):
            doctor.respond("Everyone hates me")
        self.assertEqual(doctor.respond("Everyone hates me"), first)

    def test_none_fallback_also_cycles(self):
        doctor = eliza.Doctor()
        replies = [doctor.respond("aardvark banana") for _ in range(len(rules.NONE))]
        self.assertEqual(sorted(replies), sorted(rules.NONE))


class TestMemory(unittest.TestCase):
    def test_something_said_earlier_comes_back(self):
        doctor = eliza.Doctor()
        doctor.respond("My mother hates me")
        for _ in range(6):
            reply = doctor.respond("aardvark banana")
            if "MOTHER" in reply:
                break
        else:
            self.fail("nothing from memory was ever offered")

    def test_memory_only_fills_from_the_ordered_path(self):
        # The degraded path cannot capture a run of the user's words, so it must
        # not queue anything -- a memory it cannot fill would be a stray slot.
        doctor = eliza.Doctor()
        doctor.respond_to_keywords(["MY", "MOTHER"])
        self.assertEqual(doctor._memory, [])


class TestBagMode(unittest.TestCase):
    """The degraded front end. These are the tests that matter."""

    NOUNS = ("MOTHER", "FATHER", "SISTER", "BROTHER", "WIFE", "CHILDREN")

    def test_never_emits_an_unfilled_slot(self):
        # A leftover digit on the panel is the visible form of "we shipped a
        # template the spotter could not fill". Sweep every keyword and every
        # keyword-plus-noun pair.
        doctor = eliza.Doctor()
        bags = [[k] for k in rules.RULES]
        bags += [[k, n] for k in rules.RULES for n in self.NOUNS]
        for bag in bags:
            for _ in range(3):     # rotate, so every template gets a turn
                reply = doctor.respond_to_keywords(bag, nouns=self.NOUNS)
                self.assertFalse(any(eliza.slot_of(w) is not None
                                     for w in reply.split()),
                                 "unfilled slot for %r: %r" % (bag, reply))
                self.assertNotIn("  ", reply)
                self.assertTrue(reply.strip())

    def test_phrase_templates_are_filtered_out(self):
        doctor = eliza.Doctor()
        for keyword, (rank, goto, ruleset) in rules.RULES.items():
            for pattern, templates in ruleset:
                usable = doctor._usable(templates, spotted=["MOTHER"], captured=[])
                for kind, _ in usable:
                    self.assertNotEqual(kind, rules.PHRASE,
                                        "%s leaked a PHRASE template" % keyword)

    def test_noun_template_echoes_a_spotted_noun(self):
        doctor = eliza.Doctor(priority={"MY": 8})
        replies = [doctor.respond_to_keywords(["MY", "SISTER"], nouns=self.NOUNS)
                   for _ in range(6)]
        self.assertTrue(any("SISTER" in r for r in replies), replies)

    def test_noun_template_is_skipped_when_nothing_was_spotted(self):
        # "WHY DO YOU SAY YOUR" is worse than changing the subject.
        doctor = eliza.Doctor(priority={"MY": 8})
        for _ in range(8):
            reply = doctor.respond_to_keywords(["MY"], nouns=self.NOUNS)
            self.assertFalse(reply.endswith("YOUR"), reply)
            self.assertFalse(any(eliza.slot_of(w) is not None
                                 for w in reply.split()), reply)

    def test_literal_slot_is_filled_from_the_word_class(self):
        doctor = eliza.Doctor(priority={"I": 6})
        reply = doctor.respond_to_keywords(["I", "AM", "UNHAPPY"])
        self.assertIn("UNHAPPY", reply)

    def test_copula_may_be_assumed(self):
        # A 40-word spotter will not reliably report "ARE"; refusing the best
        # rule in the script over it would be a poor trade.
        doctor = eliza.Doctor(priority={"I": 6})
        reply = doctor.respond_to_keywords(["I", "SAD"])
        self.assertIn("SAD", reply)

    def test_empty_bag_still_answers(self):
        doctor = eliza.Doctor()
        self.assertIn(doctor.respond_to_keywords([]), rules.NONE)

    def test_bag_is_order_independent(self):
        a = eliza.Doctor(priority={"MY": 8})
        b = eliza.Doctor(priority={"MY": 8})
        self.assertEqual(a.respond_to_keywords(["MY", "MOTHER"], nouns=self.NOUNS),
                         b.respond_to_keywords(["MOTHER", "MY"], nouns=self.NOUNS))


class TestDegradedAcceptance(unittest.TestCase):
    """Go/no-go for the whole spotter idea: does a bag of nouns get answered?

    A 25-word vocabulary and a perfect spotter. If these fail, the degraded
    build is a fortune-cookie machine and the project should either ship the
    display-only full-sentence version or not ship at all -- so treat a failure
    here as a product decision, not a bug to paper over.

    The trap this guards is subtle. Nouns are not keywords: DOCTOR hangs its
    family responses off keyword MY and its feeling ones off keyword I, both
    function words a spotter will never be given room for. Before the
    assumption machinery, every one of these answered with a NONE deflection
    while holding the noun in its hand.
    """

    NOUNS = ("MOTHER", "FATHER", "WIFE", "HUSBAND", "SISTER", "BROTHER",
             "CHILDREN", "WORK", "MONEY", "SLEEP", "DEATH", "LOVE")
    VOCABULARY = NOUNS + ("SAD", "HAPPY", "ANGRY", "AFRAID", "SICK", "WANT",
                          "NEED", "YES", "NO", "DREAM", "COMPUTER", "ALWAYS",
                          "SORRY")

    # Noun-bearing things somebody might plausibly say to a toy therapist.
    UTTERANCES = (
        "I hate my mother", "my brother died", "my wife thinks I work too much",
        "my children are afraid", "I need money", "work makes me sick",
        "my husband never listens", "my father was always angry",
        "I cannot sleep", "I am sad about my sister", "money is the problem",
        "I dream about death", "my mother is sick", "I want my children back",
        "love is complicated", "my sister is unhappy", "work is killing me",
        "I feel afraid at work", "my father died last year",
        "my wife wants a divorce",
    )

    def bag(self, text):
        """What a perfect spotter would hand over: recognised words, no order."""
        heard = {w.strip(",.!?;") for w in eliza.normalise(text)}
        return sorted(w for w in heard if w in self.VOCABULARY)

    def test_the_six_reported_failures(self):
        doctor = eliza.Doctor()
        for text in ("I hate my mother", "my brother died",
                     "my wife thinks I work too much", "my children are afraid",
                     "I need money", "work makes me sick"):
            spotted = self.bag(text)
            self.assertTrue(spotted, text)
            reply = doctor.respond_to_keywords(spotted, nouns=self.NOUNS)
            self.assertNotIn(reply, rules.NONE,
                             "deflected %r having heard %s" % (text, spotted))

    def test_most_noun_utterances_land(self):
        doctor = eliza.Doctor()
        landed = [t for t in self.UTTERANCES
                  if doctor.respond_to_keywords(self.bag(t), nouns=self.NOUNS)
                  not in rules.NONE]
        self.assertGreaterEqual(len(landed), 2 * len(self.UTTERANCES) // 3,
                                "only %d of %d landed" % (len(landed),
                                                          len(self.UTTERANCES)))

    def test_a_good_share_echo_the_noun_itself(self):
        # Landing is not enough -- a topical canned line is still canned. The
        # echo is the thing the noun budget was spent on.
        doctor = eliza.Doctor()
        echoed = 0
        for text in self.UTTERANCES:
            spotted = self.bag(text)
            reply = doctor.respond_to_keywords(spotted, nouns=self.NOUNS)
            bare = [w.rstrip("?.!,") for w in reply.split()]
            if any(word in bare for word in spotted):
                echoed += 1
        self.assertGreaterEqual(echoed, len(self.UTTERANCES) // 3,
                                "only %d of %d echoed" % (echoed, len(self.UTTERANCES)))

    def test_every_noun_alone_gets_a_reply(self):
        doctor = eliza.Doctor()
        for noun in self.NOUNS:
            reply = doctor.respond_to_keywords([noun], nouns=self.NOUNS)
            self.assertNotIn(reply, rules.NONE, noun)


class TestNoFunctionWordsNeeded(unittest.TestCase):
    """The degraded path must not depend on spotting MY, I, AM or ARE.

    Those are the worst words to ask a recogniser for -- one or two phonemes,
    unstressed, low energy, and coarticulated into whatever follows ("my
    mother" is often said with no boundary at all). DTW over MFCC templates is
    least accurate on exactly them, and a ~25-word budget cannot afford six
    slots for words that do not carry meaning.

    So the engine assumes them instead (_ASSUMED_IN_BAG), and this test is what
    stops that guarantee eroding: every case here is checked with the function
    words absent from the bag entirely, which is what the spotter will deliver.
    """

    NOUNS = ("MOTHER", "FATHER", "WIFE", "HUSBAND", "SISTER", "BROTHER",
             "CHILDREN", "WORK", "MONEY", "SLEEP", "DEATH", "LOVE")

    # (what a spotter that catches function words would hear, what it really
    # hears). The second is the only one that has to work.
    CASES = (
        (["I", "SAD"], ["SAD"]),
        (["MY", "MOTHER"], ["MOTHER"]),
        (["I", "MY", "BROTHER", "SICK"], ["BROTHER", "SICK"]),
        (["MY", "WIFE"], ["WIFE"]),
        (["MY", "CHILDREN", "ARE"], ["CHILDREN"]),
        (["I", "AM", "UNHAPPY"], ["UNHAPPY"]),
        (["MY", "FATHER", "IS", "SICK"], ["FATHER", "SICK"]),
        (["I", "WANT", "MY", "MONEY"], ["MONEY"]),
    )

    def test_content_words_alone_still_land(self):
        doctor = eliza.Doctor()
        landed = []
        for _, without in self.CASES:
            reply = doctor.respond_to_keywords(without, nouns=self.NOUNS)
            if reply not in rules.NONE:
                landed.append(without)
        self.assertGreaterEqual(len(landed), 6,
                                "only %d of %d landed without function words"
                                % (len(landed), len(self.CASES)))

    def test_the_function_words_add_nothing(self):
        """Spotting them should be a bonus, never a requirement."""
        for withfw, without in self.CASES:
            rich = eliza.Doctor().respond_to_keywords(withfw, nouns=self.NOUNS)
            poor = eliza.Doctor().respond_to_keywords(without, nouns=self.NOUNS)
            self.assertNotIn(poor, rules.NONE, without)
            self.assertEqual(rich, poor,
                             "%s answered differently from %s" % (withfw, without))

    def test_the_shipped_vocabulary_contains_no_function_words(self):
        # The REPL's word list is what the recogniser agent builds from, so a
        # function word creeping back in there is the regression to catch.
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
        import eliza_repl

        for word in ("MY", "YOUR", "I", "YOU", "AM", "ARE", "IS", "WAS"):
            self.assertNotIn(word, eliza_repl.VOCABULARY)


class TestNounBeatsEmotion(unittest.TestCase):
    """When a bag holds both a noun and a feeling, the noun has to win.

    Not a stylistic preference. The script's feeling rules fill their slot from
    the emotion word and never mention the noun, so ranking them first makes
    "my brother is sick" and "my children are sick" produce the same sentence
    -- and worse, "I AM SORRY TO HEAR YOU ARE SICK" tells the user *they* are
    sick when they said their brother was. The noun rules keep the subject.
    """

    NOUNS = ("MOTHER", "BROTHER", "SISTER", "WIFE", "CHILDREN", "WORK", "MONEY")

    def test_the_noun_survives(self):
        doctor = eliza.Doctor(priority={"MY": 6, "I": 2})
        for noun, feeling in (("BROTHER", "SICK"), ("MOTHER", "SAD"),
                              ("WIFE", "UNHAPPY"), ("CHILDREN", "SICK")):
            reply = doctor.respond_to_keywords([noun, feeling], nouns=self.NOUNS)
            self.assertNotIn(reply, rules.NONE)
            self.assertNotIn("YOU ARE " + feeling, reply,
                             "misattributed %s to the user" % feeling)

    def test_different_nouns_give_different_replies(self):
        # The tell that the noun is being used rather than discarded.
        doctor = eliza.Doctor(priority={"MY": 6, "I": 2})
        a = doctor.respond_to_keywords(["BROTHER", "SICK"], nouns=self.NOUNS)
        doctor = eliza.Doctor(priority={"MY": 6, "I": 2})
        b = doctor.respond_to_keywords(["MONEY", "SICK"], nouns=self.NOUNS)
        self.assertNotEqual(a, b)

    def test_a_feeling_alone_still_uses_the_feeling_rules(self):
        doctor = eliza.Doctor(priority={"MY": 6, "I": 2})
        self.assertIn("SAD", doctor.respond_to_keywords(["SAD"], nouns=self.NOUNS))


class TestVocabContract(unittest.TestCase):
    """src/vocab.py against the engine, which is a contract nothing else checks.

    The device build spends twenty-one recogniser classes, each needing three
    recorded takes. A class that ELIZA has no answer for is that whole cost for
    a deflection -- and it fails silently, looking like blandness rather than an
    error. So every class earns its slot here or it should not be recorded.
    """

    def setUp(self):
        try:
            import vocab
        except ImportError:
            self.skipTest("src/vocab.py not present")
        self.vocab = vocab

        # Every test below loops over a collection derived from vocab.py, so an
        # empty collection would make it pass while asserting nothing. Measured:
        # before this guard, deleting the entire vocabulary left all 71 tests
        # green. A test that filters its own inputs can be emptied by a change
        # to where those inputs come from, and the failure is invisible --
        # the count does not drop and nothing turns red.
        #
        # Floors, not exact counts: this is here to catch a collection
        # disappearing, not to freeze the vocabulary against editing.
        self.assertGreaterEqual(len(vocab.LABELS), 10, "vocabulary collapsed")
        self.assertGreaterEqual(len(vocab.NOUNS), 8, "noun list collapsed")
        self.assertGreaterEqual(len(vocab.FEELINGS), 2, "feeling list collapsed")
        self.assertGreaterEqual(len(vocab.TRIGGERS), 2, "trigger list collapsed")
        self.assertGreaterEqual(len(vocab.FORMS), len(vocab.LABELS))
        self.assertEqual(len(vocab.ECHO), len(vocab.LABELS))

    def test_nouns_are_echo_text_not_labels(self):
        # Passing labels would match nothing: respond_to_keywords upper-cases
        # the bag and compares against `nouns`, so lower-case labels silently
        # disable every echo in the system.
        for noun in self.vocab.NOUNS:
            self.assertEqual(noun, noun.upper())
            self.assertIn(noun, self.vocab.ECHO.values())

    def test_kinds_partition_the_vocabulary(self):
        grouped = (self.vocab.NOUNS + self.vocab.FEELINGS + self.vocab.TRIGGERS)
        self.assertEqual(sorted(grouped),
                         sorted(self.vocab.ECHO[l] for l in self.vocab.LABELS))

    def test_every_class_earns_its_slot(self):
        """Each spotted class alone must produce something better than NONE.

        There is no longer an exception. WANT used to be the documented one --
        every template behind DOCTOR's (* WANT NEED) class needs an object -- and
        it has been removed from the vocabulary for exactly that reason, so this
        now asserts what its name claims with nothing carved out.
        """
        doctor = eliza.Doctor(priority=self.vocab.PRIORITY)
        dead = []
        for label in self.vocab.LABELS:
            echo = self.vocab.ECHO[label]
            reply = doctor.respond_to_keywords([echo], nouns=self.vocab.NOUNS)
            if reply in rules.NONE:
                dead.append(label)
        self.assertEqual(dead, [], "classes that deflect: %s" % dead)

    def test_want_never_reaches_its_own_rule(self):
        """WANT is dead weight, and this records why rather than hiding it.

        WANT is injected below rather than drawn from the vocabulary, because it
        is no longer *in* the vocabulary -- it was cut on the strength of this
        test. Drawing it from LABELS would skip every combination and leave a
        test that passes while asserting nothing, which is worse than not having
        one: it is the evidence for a decision, quietly gone.

        DOCTOR's (* WANT NEED) templates all need an object -- "WHY DO YOU WANT
        4" -- so WANT says nothing alone. And whenever an object *is* spotted,
        the noun is a noun, so the possessive rules outrank it (PRIORITY puts MY
        above I) and answer instead. Both halves are individually correct and
        together they leave no case where the class can speak.

        Checked exhaustively over every bag of one to three vocabulary words
        containing WANT. If this ever starts failing, WANT has become useful and
        the comment in vocab.py calling it "first to cut" is out of date.
        """
        import itertools

        every = [self.vocab.ECHO[l] for l in self.vocab.LABELS] + ["WANT"]
        own = set()
        for pattern, templates in rules.RULES["I"][2]:
            if any(isinstance(e, tuple) and "WANT" in e for e in pattern):
                for _kind, _mood, payload in templates:
                    own.add(" ".join(w.rstrip("?.!") for w in payload.split()
                                     if not w.rstrip("?.!,").isdigit()))

        self.assertTrue(own, "the WANT/NEED rule vanished from the script")

        checked = 0
        for size in (1, 2, 3):
            for combo in itertools.combinations(every, size):
                if "WANT" not in combo:
                    continue
                checked += 1
                doctor = eliza.Doctor(priority=self.vocab.PRIORITY)
                for _ in range(8):
                    reply = doctor.respond_to_keywords(list(combo),
                                                       nouns=self.vocab.NOUNS)
                    bare = " ".join(w for w in reply.split() if w not in every)
                    self.assertNotIn(bare, own,
                                     "WANT reached its own rule for %s" % (combo,))

        # The loop *is* the assertion, so it has to be shown to have run. This
        # test went vacuous once already, when WANT left the vocabulary and the
        # filter above stopped matching anything.
        self.assertGreater(checked, 100, "only %d combinations exercised" % checked)

    def test_a_spotted_noun_suppresses_the_feeling_rules(self):
        """Deliberate, and the reason PRIORITY puts MY above I.

        Every template in DOCTOR's feeling classes is about *you* -- "I AM
        SORRY TO HEAR YOU ARE 5", "CAN YOU EXPLAIN WHAT MADE YOU 5". So when a
        noun is also in the bag there is no way to use the feeling without
        asserting it belongs to the user, and "my brother is sick" would come
        back as "I AM SORRY TO HEAR YOU ARE SICK".

        The feeling classes still earn their slots: they carry the whole reply
        whenever no noun is heard, which is most turns.
        """
        # No filter here any more: WANT was the one exception and it has been
        # cut from the vocabulary, so filtering for it would silently drop the
        # whole loop if FEELINGS were ever reduced to it alone.
        feelings = list(self.vocab.FEELINGS)
        self.assertGreaterEqual(len(feelings), 2)
        for feeling in feelings:
            alone = eliza.Doctor(priority=self.vocab.PRIORITY)
            self.assertIn(feeling,
                          alone.respond_to_keywords([feeling],
                                                    nouns=self.vocab.NOUNS),
                          "%s says nothing even alone" % feeling)
            with_noun = eliza.Doctor(priority=self.vocab.PRIORITY)
            reply = with_noun.respond_to_keywords(["MOTHER", feeling],
                                                  nouns=self.vocab.NOUNS)
            self.assertNotIn("YOU ARE " + feeling, reply)
            self.assertNotIn("TO BE " + feeling, reply)

    def test_every_noun_is_echoed_by_name(self):
        doctor = eliza.Doctor(priority=self.vocab.PRIORITY)
        for noun in self.vocab.NOUNS:
            for _ in range(6):
                reply = doctor.respond_to_keywords([noun], nouns=self.vocab.NOUNS)
                if noun in [w.rstrip("?.!,") for w in reply.split()]:
                    break
            else:
                self.fail("%s is never echoed back" % noun)

    def test_spoken_forms_map_to_a_known_class(self):
        for form in self.vocab.FORMS:
            self.assertIn(self.vocab.label_of(form), self.vocab.LABELS, form)


class TestReplPreviewsTheDevice(unittest.TestCase):
    """tools/eliza_repl.py must hear exactly what the board hears.

    The REPL is how anyone judges whether the conversation is any good without
    a board attached, and README points at it for that. So a vocabulary of its
    own is not a harmless convenience -- it is a preview of a device that does
    not exist. The copy it used to carry had drifted to 40 words against the
    device's 21 classes, and the extras were not obscure ones: NO, WANT and
    NEED were all three explicitly retired from vocab.py (NO because it is a
    homophone of KNOW and cost 82 points of recall), and SCHOOL and FRIEND had
    never been in any vocabulary the spotter was given. Every reply the REPL
    produced from one of those was a reply the board could never make.

    This is the same failure tools/enrol.py had -- a hand-written word list that
    quietly gained a retired word -- and it is pinned the same way.
    """

    def setUp(self):
        try:
            import vocab
            import eliza_repl
        except ImportError as exc:
            self.skipTest("not importable: %s" % exc)
        self.vocab = vocab
        self.repl = eliza_repl
        # Per TestVocabContract's setUp: every assertion below loops over a
        # collection derived from vocab.py, so an empty one would pass while
        # checking nothing.
        self.assertGreaterEqual(len(eliza_repl.SPOTTABLE), 10,
                                "the REPL vocabulary collapsed")

    def test_it_hears_the_spoken_forms_and_nothing_else(self):
        self.assertEqual(sorted(self.repl.SPOTTABLE),
                         sorted(f.upper() for f in self.vocab.FORMS))

    def test_the_bag_it_produces_is_the_bag_the_device_produces(self):
        # talk.Conversation.reply puts `vocab.ECHO[label]` in the bag, so these
        # are the only words the engine can ever see from a real press.
        self.assertEqual(sorted(set(self.repl.SPOTTABLE.values())),
                         sorted(set(self.vocab.ECHO.values())))

    def test_merged_classes_are_modelled_not_flattened(self):
        # SAD and SICK are one class, which is why confusing them is free. The
        # spotter returns the class; the form that was said does not survive.
        for form in ("sad", "sick"):
            self.assertEqual(self.repl.to_bag(form, self.repl.SPOTTABLE), ["SAD"])
        self.assertEqual(
            self.repl.to_bag("i am sad and sick", self.repl.SPOTTABLE), ["SAD"])

    def test_retired_and_phantom_words_are_gone(self):
        # NO, WANT and NEED were removed from the vocabulary deliberately;
        # SCHOOL and FRIEND were never in it. All five were in the old copy.
        for word in ("NO", "WANT", "NEED", "SCHOOL", "FRIEND"):
            self.assertNotIn(word, self.repl.SPOTTABLE,
                             "%s is not in the device vocabulary" % word)
            self.assertEqual(self.repl.to_bag(word.lower(),
                                              self.repl.SPOTTABLE), [])
        self.assertNotIn("SCHOOL", self.repl.NOUNS)
        self.assertNotIn("FRIEND", self.repl.NOUNS)

    def test_the_tables_are_vocab_s_own_rather_than_equal_to_them(self):
        # Identity, not equality. Two tables that happen to agree today are the
        # thing this test exists to prevent: PRIORITY still matched exactly
        # while the vocabulary beside it had drifted by 19 words.
        self.assertIs(self.repl.NOUNS, self.vocab.NOUNS)
        self.assertIs(self.repl.PRIORITY, self.vocab.PRIORITY)

    def test_an_explicit_word_list_still_overrides(self):
        # The escape hatch for "what would a bigger vocabulary buy" has to keep
        # working, and it maps each word to itself since it has no classes.
        wider = dict((w, w) for w in ("REMEMBER", "BECAUSE"))
        self.assertEqual(self.repl.to_bag("i remember because", wider),
                         ["BECAUSE", "REMEMBER"])


class TestNoFragmentsOnThePanel(unittest.TestCase):
    """No reply the device can produce may read as a truncated sentence.

    A user on the hardware reported "YOUR BROTHER" as "a sentence fragment and
    not a follow on question". They were right about the reading and the
    rendering was correct: it is DOCTOR's own bare echo, faithfully reproduced.
    What changed is the medium. On a teletype the reply scrolled past and the
    next prompt line delimited it; alone on a 200x200 panel it is the only text
    on screen, and unfinished text reads as a broken UI.

    So every reply now carries a mood and the punctuation that goes with it.
    These tests assert it against the real rule data rather than a hand-list,
    because a hand-list would not have caught the ones nobody thought of.
    """

    NOUNS = ("MOTHER", "FATHER", "WIFE", "HUSBAND", "SISTER", "BROTHER",
             "CHILDREN", "WORK", "MONEY", "SLEEP", "DEATH", "LOVE")

    def replies(self):
        """Every reply template, with its mood."""
        out = []
        for keyword, (rank, goto, ruleset) in rules.RULES.items():
            for pattern, templates in ruleset:
                for kind, mood, payload in templates:
                    if kind in (rules.GOTO, rules.NEWKEY, rules.PRE):
                        continue
                    out.append((keyword, kind, mood, payload))
        out += [("NONE", rules.CANNED, None, t) for t in rules.NONE]
        out += [("MEMORY", rules.PHRASE, m, t) for _, m, t in rules.MEMORY]
        return out

    def test_every_reply_is_terminated(self):
        found = self.replies()
        self.assertGreater(len(found), 150, "rule data collapsed")
        for keyword, kind, mood, payload in found:
            self.assertIn(payload[-1:], ("?", ".", "!"),
                          "%s: unterminated %r" % (keyword, payload))

    def test_every_reply_has_a_mood(self):
        for keyword, kind, mood, payload in self.replies():
            if keyword == "NONE":
                continue          # NONE is a plain tuple of strings
            self.assertIn(mood, (rules.QUESTION, rules.STATEMENT, rules.ECHO),
                          "%s: %r" % (keyword, payload))

    def test_mood_and_punctuation_agree(self):
        # The voice build reads mood; the panel reads the mark. They must not
        # disagree, or a rising reply gets a full stop read aloud.
        for keyword, kind, mood, payload in self.replies():
            if mood is None:
                continue
            expected = "." if mood == rules.STATEMENT else "?"
            self.assertEqual(payload[-1], expected,
                             "%s: %s but %r" % (keyword, mood, payload))

    def test_echo_templates_are_questions_not_statements(self):
        """The reported bug: a bare echo must be a prompt, never an assertion."""
        echoes = [p for _, _, m, p in self.replies() if m == rules.ECHO]
        self.assertGreaterEqual(len(echoes), 8, "ECHO class collapsed")
        for payload in echoes:
            self.assertTrue(payload.endswith("?"), payload)

    def test_the_reported_reply_reads_as_a_prompt(self):
        doctor = eliza.Doctor(priority={"MY": 6, "I": 2})
        for _ in range(8):
            reply = doctor.respond_to_keywords(["BROTHER"], nouns=self.NOUNS)
            if "BROTHER" in reply:
                self.assertTrue(reply.endswith("?") or reply.endswith("."), reply)
                if reply.rstrip("?.").endswith("BROTHER"):
                    self.assertTrue(reply.endswith("?"),
                                    "bare echo rendered as a statement: %r" % reply)
                break
        else:
            self.fail("BROTHER was never echoed")

    def test_no_device_reply_is_a_bare_frame(self):
        """A reply must never be framing with the slot missing -- "YOUR?".

        This is what an unfilled ECHO slot looks like once punctuation is
        attached, and it is invisible to a digit check because the digit is
        gone. It shipped for one commit while _plant still matched on a bare
        isdigit() and rendered "YOUR?" for every untagged noun.
        """
        doctor = eliza.Doctor(priority={"MY": 6, "I": 2})
        bags = [[k] for k in rules.RULES]
        bags += [[n] for n in self.NOUNS]
        bags += [[k, n] for k in rules.RULES for n in self.NOUNS[:4]]
        checked = 0
        for bag in bags:
            for _ in range(3):
                reply = doctor.respond_to_keywords(bag, nouns=self.NOUNS)
                checked += 1
                bare = reply.rstrip("?.!").split()
                # A one-word reply is fine when it is a wh-question -- "WHEN?"
                # after "always" is idiomatic English and a real follow-up.
                # Anything else of one word is a frame with its slot missing.
                if len(bare) == 1:
                    self.assertIn(bare[0],
                                  ("WHEN", "HOW", "WHY", "WHO", "WHERE", "WHAT"),
                                  "one-word reply %r for %s" % (reply, bag))
                    self.assertTrue(reply.endswith("?"), reply)
                # Only the possessives: a question may legitimately strand a
                # preposition ("WHAT INCIDENT ARE YOU THINKING OF?"), but a
                # determiner with nothing after it is always a missing slot.
                self.assertNotIn(bare[-1], ("YOUR", "MY"),
                                 "dangling frame %r for %s" % (reply, bag))
        self.assertGreater(checked, 200)

    def test_sentence_case_keeps_the_terminal_mark(self):
        self.assertEqual(eliza.sentence_case("YOUR BROTHER?"), "Your brother?")
        self.assertEqual(eliza.sentence_case("PLEASE GO ON."), "Please go on.")

    def test_slot_of_tolerates_punctuation(self):
        self.assertEqual(eliza.slot_of("3"), 3)
        self.assertEqual(eliza.slot_of("3?"), 3)
        self.assertEqual(eliza.slot_of("4."), 4)
        self.assertIsNone(eliza.slot_of("YOUR"))
        self.assertIsNone(eliza.slot_of("YOUR?"))


class TestReadsAsEnglish(unittest.TestCase):
    """The rest of the fragment audit: everything else the sweep turned up.

    Each of these was a template that rendered wrongly on a panel showing one
    reply and nothing else -- no user text, no history, no scrollback. The
    question mark fixed the bare echoes; these needed their own judgement.
    """

    def replies(self):
        out = []
        for keyword, (rank, goto, ruleset) in rules.RULES.items():
            for pattern, templates in ruleset:
                for kind, mood, payload in templates:
                    if kind not in (rules.GOTO, rules.NEWKEY, rules.PRE):
                        out.append((keyword, mood, payload))
        out += [("NONE", None, t) for t in rules.NONE]
        out += [("MEMORY", m, t) for _, m, t in rules.MEMORY]
        return out

    def test_a_negated_opener_is_a_question(self):
        # "DON'T YOU KNOW" is a question; "I DON'T UNDERSTAND THAT" is not, and
        # neither is the imperative "PLEASE DON'T APOLOGIZE". Position decides.
        opens = [(k, p) for k, m, p in self.replies()
                 if p.split()[0].strip(",") in ("DON'T", "CAN'T", "DOESN'T",
                                                "AREN'T", "ISN'T", "WON'T")]
        self.assertGreaterEqual(len(opens), 4, "negated openers vanished")
        for keyword, payload in opens:
            self.assertTrue(payload.endswith("?"), "%s: %r" % (keyword, payload))

    def test_a_negation_mid_sentence_is_not_a_question(self):
        for keyword, mood, payload in self.replies():
            first = payload.split()[0].strip(",")
            if first in ("I", "YOU", "PLEASE") and "N'T" in payload.upper():
                if not payload.rstrip("?.").upper().endswith(("DON'T YOU",
                                                              "ARE YOU")):
                    self.assertTrue(payload.endswith("."),
                                    "%s: %r" % (keyword, payload))

    def test_a_tag_question_is_a_question(self):
        # "YOU HAVE A PARTICULAR PERSON IN MIND, DON'T YOU" and
        # "YOU'RE NOT REALLY TALKING ABOUT ME - ARE YOU".
        tags = [(k, p) for k, m, p in self.replies()
                if p.rstrip("?.").upper().endswith(("DON'T YOU", "ARE YOU"))
                and ("," in p or "-" in p)]
        self.assertGreaterEqual(len(tags), 2, "tag questions vanished")
        for keyword, payload in tags:
            self.assertTrue(payload.endswith("?"), "%s: %r" % (keyword, payload))

    def test_a_two_sentence_reply_takes_the_mood_of_its_last_clause(self):
        # "HOW DO YOU DO. PLEASE STATE YOUR PROBLEM" ends in an imperative. The
        # wh-word in the greeting half must not make the whole thing a question.
        pairs = [(k, p) for k, m, p in self.replies() if "." in p[:-1]]
        self.assertTrue(pairs, "no multi-sentence replies left to check")
        for keyword, payload in pairs:
            self.assertTrue(payload.endswith("."), "%s: %r" % (keyword, payload))

    def test_the_greeting_is_terminated_too(self):
        self.assertIn(rules.GREETING[-1:], ("?", "."))
        self.assertEqual(eliza.sentence_case(rules.GREETING),
                         "How do you do. Please tell me your problem.")

    def test_the_scripts_typo_is_corrected(self):
        for keyword, mood, payload in self.replies():
            self.assertNotIn("APOLIGIZE", payload.upper(),
                             "%s still carries Weizenbaum's typo" % keyword)
        self.assertTrue(any("APOLOGIZE" in p.upper()
                            for _, _, p in self.replies()),
                        "the SORRY reply disappeared instead of being fixed")

    def test_proper_nouns_survive_sentence_case(self):
        self.assertEqual(eliza.sentence_case("I AM SORRY, I SPEAK ONLY ENGLISH."),
                         "I am sorry, I speak only English.")

    def test_no_reply_is_left_unterminated(self):
        for keyword, mood, payload in self.replies():
            self.assertIn(payload[-1:], ("?", ".", "!"),
                          "%s: %r" % (keyword, payload))


class TestPanelBudget(unittest.TestCase):
    """Replies must still fit the panel now that they are a character longer.

    Measured against the *real* screen.fit, with framebuf stubbed, rather than
    a reimplementation of it here. Two earlier versions of these tests wrapped
    text at 12 columns by hand and got the wrong answer, because that is not
    what fit() does:

    - fit() rejects a size outright when any single word exceeds the column
      width, before it ever counts lines;
    - and when it falls back to scale 1 the reply gets 23 columns, so an
      overflowing reply renders in *fewer* lines, not more. Counting lines
      cannot detect overflow at all. Ask about the scale.

    Overflow here means "dropped to the 8-pixel size", which is a legibility
    cliff rather than truncation -- nothing is lost, it just gets small.
    """

    SCALE_2 = 2
    MAX_LINES = 9

    @classmethod
    def setUpClass(cls):
        import types
        if "framebuf" not in sys.modules:
            sys.modules["framebuf"] = types.SimpleNamespace(
                FrameBuffer=object, MONO_HLSB=0)
        import screen
        cls.screen = screen

    def setUp(self):
        try:
            import vocab
        except ImportError:
            self.skipTest("src/vocab.py not present")
        self.vocab = vocab

    def replies(self):
        out = [p for _, _, rs in rules.RULES.values() for _, ts in rs
               for k, m, p in ts
               if k in (rules.CANNED, rules.LITERAL, rules.NOUN, rules.PHRASE)]
        return out + list(rules.NONE) + [t for _, _, t in rules.MEMORY]

    def spoken(self):
        """Only what the degraded path can put on the panel."""
        out = [p for _, _, rs in rules.RULES.values() for _, ts in rs
               for k, m, p in ts if k in rules.SPOTTABLE]
        return out + list(rules.NONE)

    def render(self, template, echo):
        filled = []
        for word in template.split():
            mark = ""
            while word[-1:] in "?.!,":
                mark = word[-1] + mark
                word = word[:-1]
            index = eliza.slot_of(word)
            filled.append((echo if index is not None else word) + mark)
        return self.screen.fit(" ".join(filled))

    def overflows(self, template, echo):
        scale, _lines = self.render(template, echo)
        return scale < self.SCALE_2

    def test_the_corpus_includes_memory(self):
        """MEMORY and NONE are separate tuples from RULES, and easy to omit.

        A sweep that walks RULES alone misses them silently. Kept even though
        it has never caught anything: it asserts that the sweep contains what
        it claims to sweep, which no other test here can say.
        """
        corpus = self.replies()
        for _pattern, _mood, template in rules.MEMORY:
            self.assertIn(template, corpus, "MEMORY missing from the sweep")
        for template in rules.NONE:
            self.assertIn(template, corpus, "NONE missing from the sweep")

    def test_every_vocabulary_noun_fits(self):
        """The echo domain is enumerable, so enumerate it.

        The device echoes exactly one spotted vocabulary noun -- twelve
        strings, not a distribution -- so there is nothing to sample and no
        stress input to choose. Two earlier versions of this test picked a
        representative phrase instead, and both pinned an artifact of the
        phrase rather than a property of the rule data.
        """
        self.assertGreaterEqual(len(self.vocab.NOUNS), 8, "noun list collapsed")
        for noun in self.vocab.NOUNS:
            for template in self.spoken():
                scale, lines = self.render(template, noun)
                self.assertEqual(scale, self.SCALE_2,
                                 "%r drops to scale 1 echoing %s" % (template, noun))
                self.assertLessEqual(len(lines), self.MAX_LINES,
                                     "%r is %d lines echoing %s"
                                     % (template, len(lines), noun))

    def test_the_device_echo_leaves_headroom(self):
        worst, worst_noun = 0, None
        for noun in self.vocab.NOUNS:
            for template in self.spoken():
                _scale, lines = self.render(template, noun)
                if len(lines) > worst:
                    worst, worst_noun = len(lines), noun
        self.assertLessEqual(worst, self.MAX_LINES - 2,
                             "only %d lines spare (worst: %s at %d lines)"
                             % (self.MAX_LINES - worst, worst_noun, worst))

    def test_a_noun_longer_than_eleven_letters_would_break_the_layout(self):
        """The constraint anyone editing vocab.NOUNS needs, and the real find.

        fit() skips a size when any word is wider than the column count, and
        the echoed noun carries the reply's question mark -- so a 12-letter
        noun is 13 characters in a 12-column line and drops the *entire* reply
        to 8-pixel text. GRANDMOTHER (11) is the longest that still renders at
        scale 2; RELATIONSHIP (12) does not.

        This is what the twelve-letter stress echo was accidentally detecting,
        and it is worth asserting directly rather than by proxy.
        """
        for noun in self.vocab.NOUNS:
            self.assertLessEqual(len(noun), 11,
                                 "%s is %d letters; 11 is the limit at 12 "
                                 "columns once the '?' is added"
                                 % (noun, len(noun)))
        self.assertEqual(self.screen.fit("YOUR GRANDMOTHER?")[0], 2)
        self.assertEqual(self.screen.fit("YOUR RELATIONSHIP?")[0], 1)

    def test_beyond_the_device_nothing_is_asserted_but_it_is_measured(self):
        """Longer echoes are reported, never asserted.

        How far past the real input the layout holds is worth knowing when
        someone proposes a longer noun. It is not worth a test that breaks when
        somebody edits an arbitrary string, which is how the previous two
        versions of this failed.
        """
        for echo in ("my mother always shouts at me",
                     "your mother and your father again"):
            over = [t for t in self.replies() if self.overflows(t, echo)]
            for template in over:
                self.assertNotIn(template, self.spoken(),
                                 "%r overflows at a phrase echo and the "
                                 "device can reach it" % template)


class TestAssumptionGuards(unittest.TestCase):
    """Assumption has to stay assumption, not invention."""

    NOUNS = ("MOTHER", "WORK")

    def test_nothing_heard_means_nothing_assumed(self):
        doctor = eliza.Doctor()
        for spotted in ([], ["ZZZZ"], ["AARDVARK", "BANANA"]):
            self.assertIn(doctor.respond_to_keywords(spotted, nouns=self.NOUNS),
                          rules.NONE, spotted)

    def test_a_high_ranked_wildcard_rule_cannot_hijack(self):
        # COMPUTER is rank 50 and its decomposition is a bare "(0)", so it
        # matches any bag whatsoever. Until _binds_a_heard_word was added, a
        # bag of {MOTHER} was answered with "DO COMPUTERS WORRY YOU".
        doctor = eliza.Doctor()
        for _ in range(6):
            reply = doctor.respond_to_keywords(["MOTHER"], nouns=self.NOUNS)
            self.assertNotIn("COMPUTER", reply)
            self.assertNotIn("MACHINE", reply)

    def test_a_real_keyword_still_wins_over_an_assumed_one(self):
        doctor = eliza.Doctor()
        reply = doctor.respond_to_keywords(["COMPUTER"], nouns=self.NOUNS)
        self.assertTrue("COMPUTER" in reply or "MACHINE" in reply, reply)

    def test_assumed_keywords_never_reach_the_ordered_path(self):
        # respond_to_words has real word order and must not start inventing
        # function words it can see are absent.
        doctor = eliza.Doctor()
        self.assertIn(doctor.respond("mother"), rules.NONE)


class TestTranscript(unittest.TestCase):
    """Regression against Weizenbaum's own published conversation.

    These lines are quoted in the 1966 CACM paper. If the engine stops
    reproducing them, something in the substitution or ranking order has moved.
    """

    def test_canonical_exchanges(self):
        doctor = eliza.Doctor()
        for said, expected in (
            ("Men are all alike", "IN WHAT WAY?"),
            ("They are always bugging us about something or other",
             "CAN YOU THINK OF A SPECIFIC EXAMPLE?"),
            ("Well, my boyfriend made me come here",
             # An ECHO template ("YOUR 3?") -- on the ordered path its slot
             # swallows a whole clause, and it is still a question.
             "YOUR BOYFRIEND MADE YOU COME HERE?"),
            ("I am unhappy", "I AM SORRY TO HEAR YOU ARE UNHAPPY."),
        ):
            self.assertEqual(doctor.respond(said), expected, said)

    def test_person_swap(self):
        # The script folds the I/you swap into its substitution table, so the
        # decompositions are written against already swapped text. Getting this
        # backwards makes ELIZA answer in the wrong person, which is the single
        # most obvious way for it to look broken.
        doctor = eliza.Doctor()
        self.assertEqual(doctor.respond("You are not very aggressive"),
                         "WHAT MAKES YOU THINK I AM NOT VERY AGGRESSIVE?")


class TestSentenceCase(unittest.TestCase):
    def test_capitalises_only_the_start_and_i(self):
        self.assertEqual(eliza.sentence_case("I AM SORRY TO HEAR YOU ARE SAD"),
                         "I am sorry to hear you are sad")
        self.assertEqual(eliza.sentence_case("WHAT MAKES YOU THINK I AM RIGHT"),
                         "What makes you think I am right")

    def test_handles_contractions(self):
        self.assertEqual(eliza.sentence_case("I'M SURE ITS NOT PLEASANT"),
                         "I'm sure its not pleasant")

    def test_empty(self):
        self.assertEqual(eliza.sentence_case(""), "")


class TestRuleData(unittest.TestCase):
    """The generated module. Guards against a bad regeneration."""

    def test_counts_match_the_published_script(self):
        templates = [t for _, _, ruleset in rules.RULES.values()
                     for _, ts in ruleset for t in ts]
        kinds = {}
        for kind, _mood, _payload in templates:
            kinds[kind] = kinds.get(kind, 0) + 1
        replies = sum(kinds.get(k, 0)
                      for k in (rules.CANNED, rules.LITERAL, rules.NOUN, rules.PHRASE))
        self.assertEqual(replies, 191)
        self.assertEqual(kinds[rules.CANNED], 79)
        self.assertEqual(kinds[rules.LITERAL], 12)
        # NOUN rose from 9 and PHRASE fell from 91 when transitive verbs
        # joined NOUN_CARRIERS: "WHY DO YOU WANT 4" takes a bare noun as
        # happily as "WHAT ABOUT 5" does. Echo capacity is the scarce resource,
        # so that trade is the one worth watching in this number.
        self.assertEqual(kinds[rules.NOUN], 16)
        self.assertEqual(kinds[rules.PHRASE], 84)
        self.assertEqual(len(rules.NONE), 4)
        self.assertEqual(len(rules.MEMORY), 4)
        self.assertEqual(len(rules.SUBS), 18)

    def test_every_goto_target_exists(self):
        for keyword, (rank, goto, ruleset) in rules.RULES.items():
            if goto:
                self.assertIn(goto, rules.RULES, keyword)
            for _, templates in ruleset:
                for kind, _mood, payload in templates:
                    if kind == rules.GOTO:
                        self.assertIn(payload, rules.RULES, keyword)
                    elif kind == rules.PRE:
                        self.assertIn(payload[1], rules.RULES, keyword)

    def test_every_slot_refers_to_a_real_component(self):
        for keyword, (rank, goto, ruleset) in rules.RULES.items():
            for pattern, templates in ruleset:
                for kind, _mood, payload in templates:
                    if kind in (rules.GOTO, rules.NEWKEY, rules.PRE):
                        continue
                    for word in payload.split():
                        index = eliza.slot_of(word)
                        if index is not None:
                            self.assertLessEqual(index, len(pattern),
                                                 "%s: %r" % (keyword, payload))

    def test_classification_is_self_consistent(self):
        # A CANNED template must have no slots; a PHRASE one must have at least
        # one that points at a wildcard. Cheap, and it would have caught the
        # bug where control forms were being counted as canned replies.
        for keyword, (rank, goto, ruleset) in rules.RULES.items():
            for pattern, templates in ruleset:
                for kind, _mood, payload in templates:
                    if kind in (rules.GOTO, rules.NEWKEY, rules.PRE):
                        continue          # control forms carry no reply text
                    slots = [eliza.slot_of(w) for w in payload.split()]
                    slots = [i for i in slots if i is not None]
                    if kind == rules.CANNED:
                        self.assertFalse(slots, "%s: %r" % (keyword, payload))
                    elif kind in (rules.LITERAL, rules.NOUN, rules.PHRASE):
                        self.assertTrue(slots, "%s: %r" % (keyword, payload))

    def test_tagged_words_cover_the_family_list(self):
        family = [w for w, tags in rules.TAGS.items() if "FAMILY" in tags]
        for word in ("MOTHER", "FATHER", "SISTER", "BROTHER", "WIFE", "CHILDREN"):
            self.assertIn(word, family)


class TestMicroPythonCompatibility(unittest.TestCase):
    """The engine has to import on a board that has no `re` and no `random`."""

    FORBIDDEN = ("re", "random", "collections", "itertools", "typing",
                 "dataclasses", "enum", "functools")

    def read(self, name):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "src", name)
        with open(path) as handle:
            return handle.read()

    def test_no_host_only_imports(self):
        for name in ("eliza.py", "eliza_rules.py"):
            for line in self.read(name).split("\n"):
                line = line.strip()
                if not line.startswith(("import ", "from ")):
                    continue
                module = line.split()[1].split(".")[0]
                self.assertNotIn(module, self.FORBIDDEN,
                                 "%s imports %s" % (name, module))

    def test_engine_imports_only_os_and_its_rules(self):
        imported = set()
        for line in self.read("eliza.py").split("\n"):
            line = line.strip()
            if line.startswith("import "):
                imported.add(line.split()[1].split(".")[0])
        self.assertEqual(imported, {"os", "eliza_rules"})

    def test_cross_compiles_for_micropython(self):
        """The real check: MicroPython's own compiler accepts both modules.

        Skipped when mpy-cross is not reachable, so a network-less run still
        passes -- but this is the test that would actually catch a construct
        CPython allows and the board does not.
        """
        import shutil
        import subprocess
        import tempfile

        if shutil.which("uvx") is None:
            self.skipTest("uvx not installed")
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
        with tempfile.TemporaryDirectory() as out:
            for name in ("eliza.py", "eliza_rules.py"):
                try:
                    done = subprocess.run(
                        ["uvx", "--quiet", "mpy-cross", os.path.join(src, name),
                         "-o", os.path.join(out, name + ".mpy")],
                        capture_output=True, timeout=120)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    self.skipTest("mpy-cross unavailable (%s)" % exc)
                self.assertEqual(done.returncode, 0,
                                 "%s: %s" % (name, done.stderr.decode()))

    def test_no_f_strings(self):
        # The repo formats with %, and MicroPython's f-string support is
        # partial; staying with % keeps one less thing to discover on device.
        for name in ("eliza.py", "eliza_rules.py"):
            self.assertNotIn('f"', self.read(name))


class TestRobustness(unittest.TestCase):
    def test_never_raises_on_arbitrary_input(self):
        doctor = eliza.Doctor()
        for text in ("", "   ", ",,,", "?", "BUT", "but but but",
                     "I", "my my my my", "COMPUTER COMPUTER",
                     "éèê", "a" * 300,
                     " ".join(["I", "AM", "SAD"] * 40)):
            reply = doctor.respond(text)
            self.assertTrue(isinstance(reply, str) and reply)

    def test_never_raises_on_arbitrary_bags(self):
        doctor = eliza.Doctor()
        for bag in ([], [""], ["ZZZZ"], list(rules.RULES),
                    ["MY"] * 20, ["mother", "my"]):
            reply = doctor.respond_to_keywords([w for w in bag if w])
            self.assertTrue(isinstance(reply, str) and reply)


if __name__ == "__main__":
    unittest.main(verbosity=2)
