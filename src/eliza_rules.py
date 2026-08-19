"""Weizenbaum's 1966 DOCTOR script as Python data. Generated -- do not edit.

Regenerate with `uv run tools/make_eliza_rules.py`. The interesting
decisions (which template is fillable from what) live in that script, not
here; edit it and re-run rather than patching this file.

Reply kinds tag each template with what it needs in order to be spoken:

    CANNED   no slots -- always usable
    LITERAL  slots refer to a word the decomposition pinned down, so a
             keyword spotter that heard that word can fill them
    NOUN     slots refer to the user's own words, but sit where a bare
             noun is grammatical, so a spotted noun can stand in
    PHRASE   slots need a real clause. Reachable only from a transcript.

A bag-of-keywords front end can use CANNED, LITERAL and NOUN. Only a full
transcript unlocks PHRASE. That split is the whole reason this is data.

Control kinds carry no reply text of their own:

    GOTO     ("G", "WHAT")             answer as if WHAT had matched
    NEWKEY   ("K", None)               abandon this keyword, try the next
    PRE      ("P", ("I ARE 3", "YOU")) rewrite the input, then goto

Every reply also carries a mood, and the terminal punctuation that goes
with it. The 1966 script has none: on a teletype the next prompt line
delimited a reply, but alone on a 200x200 panel the same text reads as
unfinished, and the short ones read as broken.

    QUESTION   a full question. Rising. Gets "?"
    STATEMENT  a complete sentence. Falling. Gets "."
    ECHO       the user's own word handed back as a prompt. Gets "?"

ECHO is separate from QUESTION because it is the case that reads as a
fragment -- "YOUR 4", "REALLY, 2" -- and because spoken it wants a
shorter, higher rise than a full question. Control forms carry None.
"""

CANNED = "C"
LITERAL = "L"
NOUN = "N"
PHRASE = "T"

QUESTION = "Q"
STATEMENT = "S"
ECHO = "E"

GOTO = "G"
NEWKEY = "K"
PRE = "P"

# Kinds a reply can actually be built from without a transcript.
SPOTTABLE = (CANNED, LITERAL, NOUN)

GREETING = 'HOW DO YOU DO. PLEASE TELL ME YOUR PROBLEM'

# Applied to every input word before anything else. This is also ELIZA's
# person swap -- I/YOU, MY/YOUR -- which is why the decompositions read as
# though the user had said them about the doctor.
SUBS = {
    'AM': 'ARE',
    'CANT': "CAN'T",
    'DAD': 'FATHER',
    'DONT': "DON'T",
    'DREAMED': 'DREAMT',
    'DREAMS': 'DREAM',
    'I': 'YOU',
    "I'M": "YOU'RE",
    'ME': 'YOU',
    'MOM': 'MOTHER',
    'MY': 'YOUR',
    'MYSELF': 'YOURSELF',
    'WERE': 'WAS',
    'WONT': "WON'T",
    'YOU': 'I',
    "YOU'RE": "I'M",
    'YOUR': 'MY',
    'YOURSELF': 'MYSELF',
}

# The script's DLIST tags, referenced from decompositions as "/FAMILY".
TAGS = {
    'BELIEVE': ('BELIEF',),
    'BROTHER': ('FAMILY',),
    'CHILDREN': ('FAMILY',),
    'DAD': ('FAMILY',),
    'FATHER': ('NOUN', 'FAMILY'),
    'FEEL': ('BELIEF',),
    'HUSBAND': ('FAMILY',),
    'MOM': ('FAMILY',),
    'MOTHER': ('NOUN', 'FAMILY'),
    'SISTER': ('FAMILY',),
    'THINK': ('BELIEF',),
    'WIFE': ('FAMILY',),
    'WISH': ('BELIEF',),
}

# keyword -> (rank, goto, ((decomposition, ((kind, mood, payload), ...)), ...))
#
# Rank arbitrates when the input contains several keywords; the script
# gives COMPUTER 50 so that talking about machines wins over anything else.
RULES = {
    'ALIKE': (10, 'DIT', (
    )),
    'ALWAYS': (1, None, (
        ((0,), (
            ('C', 'Q', 'CAN YOU THINK OF A SPECIFIC EXAMPLE?'),
            ('C', 'Q', 'WHEN?'),
            ('C', 'Q', 'WHAT INCIDENT ARE YOU THINKING OF?'),
            ('C', 'E', 'REALLY, ALWAYS?'),
        )),
    )),
    'AM': (0, None, (
        ((0, 'ARE', 'YOU', 0), (
            ('T', 'Q', 'DO YOU BELIEVE YOU ARE 4?'),
            ('T', 'Q', 'WOULD YOU WANT TO BE 4?'),
            ('T', 'S', 'YOU WISH I WOULD TELL YOU YOU ARE 4.'),
            ('T', 'Q', 'WHAT WOULD IT MEAN IF YOU WERE 4?'),
            ('G', None, 'WHAT'),
        )),
        ((0,), (
            ('C', 'Q', "WHY DO YOU SAY 'AM'?"),
            ('C', 'S', "I DON'T UNDERSTAND THAT."),
        )),
    )),
    'ARE': (0, None, (
        ((0, 'ARE', 'I', 0), (
            ('T', 'Q', 'WHY ARE YOU INTERESTED IN WHETHER I AM 4 OR NOT?'),
            ('T', 'Q', "WOULD YOU PREFER IF I WEREN'T 4?"),
            ('T', 'S', 'PERHAPS I AM 4 IN YOUR FANTASIES.'),
            ('T', 'Q', 'DO YOU SOMETIMES THINK I AM 4?'),
            ('G', None, 'WHAT'),
        )),
        ((0, 'ARE', 0), (
            ('T', 'Q', 'DID YOU THINK THEY MIGHT NOT BE 3?'),
            ('T', 'Q', 'WOULD YOU LIKE IT IF THEY WERE NOT 3?'),
            ('T', 'Q', 'WHAT IF THEY WERE NOT 3?'),
            ('T', 'S', 'POSSIBLY THEY ARE 3.'),
        )),
    )),
    'BECAUSE': (0, None, (
        ((0,), (
            ('C', 'Q', 'IS THAT THE REAL REASON?'),
            ('C', 'S', "DON'T ANY OTHER REASONS COME TO MIND."),
            ('C', 'Q', 'DOES THAT REASON SEEM TO EXPLAIN ANYTHING ELSE?'),
            ('C', 'Q', 'WHAT OTHER REASONS MIGHT THERE BE?'),
        )),
    )),
    'CAN': (0, None, (
        ((0, 'CAN', 'I', 0), (
            ('T', 'S', "YOU BELIEVE I CAN 4 DON'T YOU."),
            ('G', None, 'WHAT'),
            ('T', 'S', 'YOU WANT ME TO BE ABLE TO 4.'),
            ('T', 'S', 'PERHAPS YOU WOULD LIKE TO BE ABLE TO 4 YOURSELF.'),
        )),
        ((0, 'CAN', 'YOU', 0), (
            ('T', 'Q', 'WHETHER OR NOT YOU CAN 4 DEPENDS ON YOU MORE THAN ON ME?'),
            ('T', 'Q', 'DO YOU WANT TO BE ABLE TO 4?'),
            ('T', 'S', "PERHAPS YOU DON'T WANT TO 4."),
            ('G', None, 'WHAT'),
        )),
    )),
    'CERTAINLY': (0, 'YES', (
    )),
    'COMPUTER': (50, None, (
        ((0,), (
            ('C', 'Q', 'DO COMPUTERS WORRY YOU?'),
            ('C', 'Q', 'WHY DO YOU MENTION COMPUTERS?'),
            ('C', 'Q', 'WHAT DO YOU THINK MACHINES HAVE TO DO WITH YOUR PROBLEM?'),
            ('C', 'S', "DON'T YOU THINK COMPUTERS CAN HELP PEOPLE."),
            ('C', 'Q', 'WHAT ABOUT MACHINES WORRIES YOU?'),
            ('C', 'Q', 'WHAT DO YOU THINK ABOUT MACHINES?'),
        )),
    )),
    'COMPUTERS': (50, 'COMPUTER', (
    )),
    'DEUTSCH': (0, 'XFREMD', (
    )),
    'DIT': (0, None, (
        ((0,), (
            ('C', 'Q', 'IN WHAT WAY?'),
            ('C', 'Q', 'WHAT RESEMBLANCE DO YOU SEE?'),
            ('C', 'Q', 'WHAT DOES THAT SIMILARITY SUGGEST TO YOU?'),
            ('C', 'Q', 'WHAT OTHER CONNECTIONS DO YOU SEE?'),
            ('C', 'Q', 'WHAT DO YOU SUPPOSE THAT RESEMBLANCE MEANS?'),
            ('C', 'Q', 'WHAT IS THE CONNECTION, DO YOU SUPPOSE?'),
            ('C', 'Q', 'COULD THERE REALLY BE SOME CONNECTION?'),
            ('C', 'Q', 'HOW?'),
        )),
    )),
    'DREAM': (3, None, (
        ((0,), (
            ('C', 'Q', 'WHAT DOES THAT DREAM SUGGEST TO YOU?'),
            ('C', 'Q', 'DO YOU DREAM OFTEN?'),
            ('C', 'Q', 'WHAT PERSONS APPEAR IN YOUR DREAMS?'),
            ('C', 'S', "DON'T YOU BELIEVE THAT DREAM HAS SOMETHING TO DO WITH YOUR PROBLEM."),
            ('K', None, None),
        )),
    )),
    'DREAMED': (4, 'DREAMT', (
    )),
    'DREAMS': (3, 'DREAM', (
    )),
    'DREAMT': (4, None, (
        ((0, 'YOU', 'DREAMT', 0), (
            ('T', 'E', 'REALLY, 4?'),
            ('T', 'Q', 'HAVE YOU EVER FANTASIED 4 WHILE YOU WERE AWAKE?'),
            ('T', 'Q', 'HAVE YOU DREAMT 4 BEFORE?'),
            ('G', None, 'DREAM'),
            ('K', None, None),
        )),
    )),
    'ESPANOL': (0, 'XFREMD', (
    )),
    'EVERYONE': (2, None, (
        ((0, ('EVERYONE', 'EVERYBODY', 'NOBODY', 'NOONE'), 0), (
            ('L', 'E', 'REALLY, 2?'),
            ('L', 'E', 'SURELY NOT 2?'),
            ('C', 'Q', 'CAN YOU THINK OF ANYONE IN PARTICULAR?'),
            ('C', 'Q', 'WHO, FOR EXAMPLE?'),
            ('C', 'S', 'YOU ARE THINKING OF A VERY SPECIAL PERSON.'),
            ('C', 'Q', 'WHO, MAY I ASK?'),
            ('C', 'Q', 'SOMEONE SPECIAL PERHAPS?'),
            ('C', 'S', "YOU HAVE A PARTICULAR PERSON IN MIND, DON'T YOU."),
            ('C', 'Q', "WHO DO YOU THINK YOU'RE TALKING ABOUT?"),
        )),
    )),
    'FRANCAIS': (0, 'XFREMD', (
    )),
    'HELLO': (0, None, (
        ((0,), (
            ('C', 'Q', 'HOW DO YOU DO. PLEASE STATE YOUR PROBLEM?'),
        )),
    )),
    'HOW': (0, 'WHAT', (
    )),
    'I': (0, None, (
        ((0, 'YOU', ('WANT', 'NEED'), 0), (
            ('N', 'Q', 'WHAT WOULD IT MEAN TO YOU IF YOU GOT 4?'),
            ('N', 'Q', 'WHY DO YOU WANT 4?'),
            ('N', 'Q', 'SUPPOSE YOU GOT 4 SOON?'),
            ('N', 'Q', 'WHAT IF YOU NEVER GOT 4?'),
            ('T', 'Q', 'WHAT WOULD GETTING 4 MEAN TO YOU?'),
            ('N', 'Q', 'WHAT DOES WANTING 4 HAVE TO DO WITH THIS DISCUSSION?'),
        )),
        ((0, 'YOU', 'ARE', 0, ('SAD', 'UNHAPPY', 'DEPRESSED', 'SICK', 'ANGRY', 'AFRAID', 'LONELY', 'ANXIOUS'), 0), (
            ('L', 'S', 'I AM SORRY TO HEAR YOU ARE 5.'),
            ('L', 'Q', 'DO YOU THINK COMING HERE WILL HELP YOU NOT TO BE 5?'),
            ('L', 'S', "I'M SURE ITS NOT PLEASANT TO BE 5."),
            ('L', 'Q', 'CAN YOU EXPLAIN WHAT MADE YOU 5?'),
        )),
        ((0, 'YOU', 'ARE', 0, ('HAPPY', 'ELATED', 'GLAD', 'BETTER'), 0), (
            ('L', 'Q', 'HOW HAVE I HELPED YOU TO BE 5?'),
            ('L', 'Q', 'HAS YOUR TREATMENT MADE YOU 5?'),
            ('L', 'Q', 'WHAT MAKES YOU 5 JUST NOW?'),
            ('L', 'Q', 'CAN YOU EXPLAIN WHY YOU ARE SUDDENLY 5?'),
        )),
        ((0, 'YOU', 'WAS', 0), (
            ('G', None, 'WAS'),
        )),
        ((0, 'YOU', '/BELIEF', 'YOU', 0), (
            ('C', 'Q', 'DO YOU REALLY THINK SO?'),
            ('T', 'S', 'BUT YOU ARE NOT SURE YOU 5.'),
            ('T', 'Q', 'DO YOU REALLY DOUBT YOU 5?'),
        )),
        ((0, 'YOU', 0, '/BELIEF', 0, 'I', 0), (
            ('G', None, 'YOU'),
        )),
        ((0, 'YOU', 'ARE', 0), (
            ('T', 'Q', 'IS IT BECAUSE YOU ARE 4 THAT YOU CAME TO ME?'),
            ('T', 'Q', 'HOW LONG HAVE YOU BEEN 4?'),
            ('T', 'Q', 'DO YOU BELIEVE IT NORMAL TO BE 4?'),
            ('T', 'Q', 'DO YOU ENJOY BEING 4?'),
        )),
        ((0, 'YOU', ("CAN'T", 'CANNOT'), 0), (
            ('T', 'Q', "HOW DO YOU KNOW YOU CAN'T 4?"),
            ('C', 'Q', 'HAVE YOU TRIED?'),
            ('T', 'S', 'PERHAPS YOU COULD 4 NOW.'),
            ('T', 'Q', 'DO YOU REALLY WANT TO BE ABLE TO 4?'),
        )),
        ((0, 'YOU', "DON'T", 0), (
            ('T', 'S', "DON'T YOU REALLY 4."),
            ('T', 'Q', "WHY DON'T YOU 4?"),
            ('T', 'Q', 'DO YOU WISH TO BE ABLE TO 4?'),
            ('C', 'Q', 'DOES THAT TROUBLE YOU?'),
        )),
        ((0, 'YOU', 'FEEL', 0), (
            ('C', 'S', 'TELL ME MORE ABOUT SUCH FEELINGS.'),
            ('T', 'Q', 'DO YOU OFTEN FEEL 4?'),
            ('T', 'Q', 'DO YOU ENJOY FEELING 4?'),
            ('T', 'Q', 'OF WHAT DOES FEELING 4 REMIND YOU?'),
        )),
        ((0, 'YOU', 0, 'I', 0), (
            ('T', 'S', 'PERHAPS IN YOUR FANTASY WE 3 EACH OTHER.'),
            ('T', 'Q', 'DO YOU WISH TO 3 ME?'),
            ('T', 'S', 'YOU SEEM TO NEED TO 3 ME.'),
            ('T', 'Q', 'DO YOU 3 ANYONE ELSE?'),
        )),
        ((0,), (
            ('T', 'S', 'YOU SAY 1.'),
            ('C', 'Q', 'CAN YOU ELABORATE ON THAT?'),
            ('T', 'Q', 'DO YOU SAY 1 FOR SOME SPECIAL REASON?'),
            ('C', 'S', "THAT'S QUITE INTERESTING."),
        )),
    )),
    "I'M": (0, None, (
        ((0, "YOU'RE", 0), (
            ('P', None, ('YOU ARE 3', 'I')),
        )),
    )),
    'IF': (3, None, (
        ((0, 'IF', 0), (
            ('T', 'Q', 'DO YOU THINK ITS LIKELY THAT 3?'),
            ('T', 'Q', 'DO YOU WISH THAT 3?'),
            ('N', 'Q', 'WHAT DO YOU THINK ABOUT 3?'),
            ('T', 'E', 'REALLY, 2 3?'),
        )),
    )),
    'ITALIANO': (0, 'XFREMD', (
    )),
    'LIKE': (10, None, (
        ((0, ('AM', 'IS', 'ARE', 'WAS'), 0, 'LIKE', 0), (
            ('G', None, 'DIT'),
        )),
        ((0,), (
            ('K', None, None),
        )),
    )),
    'MACHINE': (50, 'COMPUTER', (
    )),
    'MACHINES': (50, 'COMPUTER', (
    )),
    'MAYBE': (0, 'PERHAPS', (
    )),
    'MY': (2, None, (
        ((0, 'YOUR', 0, '/FAMILY', 0), (
            ('C', 'S', 'TELL ME MORE ABOUT YOUR FAMILY.'),
            ('T', 'Q', 'WHO ELSE IN YOUR FAMILY 5?'),
            ('L', 'E', 'YOUR 4?'),
            ('L', 'Q', 'WHAT ELSE COMES TO MIND WHEN YOU THINK OF YOUR 4?'),
        )),
        ((0, 'YOUR', 0), (
            ('N', 'E', 'YOUR 3?'),
            ('N', 'Q', 'WHY DO YOU SAY YOUR 3?'),
            ('C', 'Q', 'DOES THAT SUGGEST ANYTHING ELSE WHICH BELONGS TO YOU?'),
            ('T', 'Q', 'IS IT IMPORTANT TO YOU THAT 2 3?'),
        )),
    )),
    'NAME': (15, None, (
        ((0,), (
            ('C', 'S', 'I AM NOT INTERESTED IN NAMES.'),
            ('C', 'S', "I'VE TOLD YOU BEFORE, I DON'T CARE ABOUT NAMES - PLEASE CONTINUE."),
        )),
    )),
    'NO': (0, None, (
        ((0,), (
            ('C', 'Q', "ARE YOU SAYING 'NO' JUST TO BE NEGATIVE?"),
            ('C', 'S', 'YOU ARE BEING A BIT NEGATIVE.'),
            ('C', 'Q', 'WHY NOT?'),
            ('C', 'Q', "WHY 'NO'?"),
        )),
    )),
    'PERHAPS': (0, None, (
        ((0,), (
            ('C', 'S', "YOU DON'T SEEM QUITE CERTAIN."),
            ('C', 'Q', 'WHY THE UNCERTAIN TONE?'),
            ('C', 'S', "CAN'T YOU BE MORE POSITIVE."),
            ('C', 'S', "YOU AREN'T SURE."),
            ('C', 'S', "DON'T YOU KNOW."),
        )),
    )),
    'REMEMBER': (5, None, (
        ((0, 'YOU', 'REMEMBER', 0), (
            ('N', 'Q', 'DO YOU OFTEN THINK OF 4?'),
            ('N', 'Q', 'DOES THINKING OF 4 BRING ANYTHING ELSE TO MIND?'),
            ('C', 'Q', 'WHAT ELSE DO YOU REMEMBER?'),
            ('N', 'Q', 'WHY DO YOU REMEMBER 4 JUST NOW?'),
            ('N', 'Q', 'WHAT IN THE PRESENT SITUATION REMINDS YOU OF 4?'),
            ('T', 'Q', 'WHAT IS THE CONNECTION BETWEEN ME AND 4?'),
        )),
        ((0, 'DO', 'I', 'REMEMBER', 0), (
            ('N', 'Q', 'DID YOU THINK I WOULD FORGET 5?'),
            ('T', 'Q', 'WHY DO YOU THINK I SHOULD RECALL 5 NOW?'),
            ('N', 'Q', 'WHAT ABOUT 5?'),
            ('G', None, 'WHAT'),
            ('T', 'S', 'YOU MENTIONED 5.'),
        )),
        ((0,), (
            ('K', None, None),
        )),
    )),
    'SAME': (10, 'DIT', (
    )),
    'SORRY': (0, None, (
        ((0,), (
            ('C', 'S', "PLEASE DON'T APOLIGIZE."),
            ('C', 'S', 'APOLOGIES ARE NOT NECESSARY.'),
            ('C', 'Q', 'WHAT FEELINGS DO YOU HAVE WHEN YOU APOLOGIZE?'),
            ('C', 'S', "I'VE TOLD YOU THAT APOLOGIES ARE NOT REQUIRED."),
        )),
    )),
    'WAS': (2, None, (
        ((0, 'WAS', 'YOU', 0), (
            ('T', 'Q', 'WHAT IF YOU WERE 4?'),
            ('T', 'Q', 'DO YOU THINK YOU WERE 4?'),
            ('T', 'Q', 'WERE YOU 4?'),
            ('T', 'Q', 'WHAT WOULD IT MEAN IF YOU WERE 4?'),
            ('T', 'Q', "WHAT DOES ' 4 ' SUGGEST TO YOU?"),
            ('G', None, 'WHAT'),
        )),
        ((0, 'YOU', 'WAS', 0), (
            ('C', 'Q', 'WERE YOU REALLY?'),
            ('T', 'Q', 'WHY DO YOU TELL ME YOU WERE 4 NOW?'),
            ('T', 'S', 'PERHAPS I ALREADY KNEW YOU WERE 4.'),
        )),
        ((0, 'WAS', 'I', 0), (
            ('T', 'Q', 'WOULD YOU LIKE TO BELIEVE I WAS 4?'),
            ('T', 'Q', 'WHAT SUGGESTS THAT I WAS 4?'),
            ('C', 'Q', 'WHAT DO YOU THINK?'),
            ('T', 'S', 'PERHAPS I WAS 4.'),
            ('T', 'Q', 'WHAT IF I HAD BEEN 4?'),
        )),
        ((0,), (
            ('K', None, None),
        )),
    )),
    'WERE': (0, 'WAS', (
    )),
    'WHAT': (0, None, (
        ((0,), (
            ('C', 'Q', 'WHY DO YOU ASK?'),
            ('C', 'Q', 'DOES THAT QUESTION INTEREST YOU?'),
            ('C', 'Q', 'WHAT IS IT YOU REALLY WANT TO KNOW?'),
            ('C', 'Q', 'ARE SUCH QUESTIONS MUCH ON YOUR MIND?'),
            ('C', 'Q', 'WHAT ANSWER WOULD PLEASE YOU MOST?'),
            ('C', 'Q', 'WHAT DO YOU THINK?'),
            ('C', 'Q', 'WHAT COMES TO YOUR MIND WHEN YOU ASK THAT?'),
            ('C', 'Q', 'HAVE YOU ASKED SUCH QUESTIONS BEFORE?'),
            ('C', 'Q', 'HAVE YOU ASKED ANYONE ELSE?'),
        )),
    )),
    'WHEN': (0, 'WHAT', (
    )),
    'WHY': (0, 'WHAT', (
        ((0, 'WHY', "DON'T", 'I', 0), (
            ('T', 'Q', "DO YOU BELIEVE I DON'T 5?"),
            ('T', 'S', 'PERHAPS I WILL 5 IN GOOD TIME.'),
            ('T', 'Q', 'SHOULD YOU 5 YOURSELF?'),
            ('T', 'S', 'YOU WANT ME TO 5.'),
            ('G', None, 'WHAT'),
        )),
        ((0, 'WHY', "CAN'T", 'YOU', 0), (
            ('T', 'Q', 'DO YOU THINK YOU SHOULD BE ABLE TO 5?'),
            ('T', 'Q', 'DO YOU WANT TO BE ABLE TO 5?'),
            ('T', 'Q', 'DO YOU BELIEVE THIS WILL HELP YOU TO 5?'),
            ('T', 'Q', "HAVE YOU ANY IDEA WHY YOU CAN'T 5?"),
            ('G', None, 'WHAT'),
        )),
    )),
    'XFREMD': (0, None, (
        ((0,), (
            ('C', 'S', 'I AM SORRY, I SPEAK ONLY ENGLISH.'),
        )),
    )),
    'YES': (0, None, (
        ((0,), (
            ('C', 'S', 'YOU SEEM QUITE POSITIVE.'),
            ('C', 'S', 'YOU ARE SURE.'),
            ('C', 'S', 'I SEE.'),
            ('C', 'S', 'I UNDERSTAND.'),
        )),
    )),
    'YOU': (0, None, (
        ((0, 'I', 'REMIND', 'YOU', 'OF', 0), (
            ('G', None, 'DIT'),
        )),
        ((0, 'I', 'ARE', 0), (
            ('T', 'Q', 'WHAT MAKES YOU THINK I AM 4?'),
            ('T', 'Q', 'DOES IT PLEASE YOU TO BELIEVE I AM 4?'),
            ('T', 'Q', 'DO YOU SOMETIMES WISH YOU WERE 4?'),
            ('T', 'S', 'PERHAPS YOU WOULD LIKE TO BE 4.'),
        )),
        ((0, 'I', 0, 'YOU'), (
            ('T', 'Q', 'WHY DO YOU THINK I 3 YOU?'),
            ('T', 'S', "YOU LIKE TO THINK I 3 YOU - DON'T YOU."),
            ('T', 'Q', 'WHAT MAKES YOU THINK I 3 YOU?'),
            ('T', 'E', 'REALLY, I 3 YOU?'),
            ('T', 'Q', 'DO YOU WISH TO BELIEVE I 3 YOU?'),
            ('T', 'Q', 'SUPPOSE I DID 3 YOU - WHAT WOULD THAT MEAN?'),
            ('T', 'Q', 'DOES SOMEONE ELSE BELIEVE I 3 YOU?'),
        )),
        ((0, 'I', 0), (
            ('C', 'S', 'WE WERE DISCUSSING YOU - NOT ME.'),
            ('T', 'E', 'OH, I 3?'),
            ('C', 'S', "YOU'RE NOT REALLY TALKING ABOUT ME - ARE YOU."),
            ('C', 'Q', 'WHAT ARE YOUR FEELINGS NOW?'),
        )),
    )),
    "YOU'RE": (0, None, (
        ((0, "I'M", 0), (
            ('P', None, ('I ARE 3', 'YOU')),
        )),
    )),
    'YOUR': (0, None, (
        ((0, 'MY', 0), (
            ('N', 'Q', 'WHY ARE YOU CONCERNED OVER MY 3?'),
            ('T', 'Q', 'WHAT ABOUT YOUR OWN 3?'),
            ('T', 'Q', 'ARE YOU WORRIED ABOUT SOMEONE ELSES 3?'),
            ('N', 'E', 'REALLY, MY 3?'),
        )),
    )),
}

# The MEMORY queue: when the input contains no keyword at all, ELIZA may
# answer something the user said several turns ago instead of admitting it
# understood nothing. It is the single cheapest trick in the script.
MEMORY_KEYWORD = 'MY'
MEMORY = (
    ((0, 'YOUR', 0), 'Q', 'LETS DISCUSS FURTHER WHY YOUR 3?'),
    ((0, 'YOUR', 0), 'S', 'EARLIER YOU SAID YOUR 3.'),
    ((0, 'YOUR', 0), 'E', 'BUT YOUR 3?'),
    ((0, 'YOUR', 0), 'Q', 'DOES THAT HAVE ANYTHING TO DO WITH THE FACT THAT YOUR 3?'),
)

# Last resort, when there is no keyword and nothing in memory.
NONE = (
    'I AM NOT SURE I UNDERSTAND YOU FULLY.',
    'PLEASE GO ON.',
    'WHAT DOES THAT SUGGEST TO YOU?',
    'DO YOU FEEL STRONGLY ABOUT DISCUSSING SUCH THINGS?',
)
