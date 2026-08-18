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
"""

CANNED = "C"
LITERAL = "L"
NOUN = "N"
PHRASE = "T"

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

# keyword -> (rank, goto, ((decomposition, ((kind, payload), ...)), ...))
#
# Rank arbitrates when the input contains several keywords; the script
# gives COMPUTER 50 so that talking about machines wins over anything else.
RULES = {
    'ALIKE': (10, 'DIT', (
    )),
    'ALWAYS': (1, None, (
        ((0,), (
            ('C', 'CAN YOU THINK OF A SPECIFIC EXAMPLE'),
            ('C', 'WHEN'),
            ('C', 'WHAT INCIDENT ARE YOU THINKING OF'),
            ('C', 'REALLY, ALWAYS'),
        )),
    )),
    'AM': (0, None, (
        ((0, 'ARE', 'YOU', 0), (
            ('T', 'DO YOU BELIEVE YOU ARE 4'),
            ('T', 'WOULD YOU WANT TO BE 4'),
            ('T', 'YOU WISH I WOULD TELL YOU YOU ARE 4'),
            ('T', 'WHAT WOULD IT MEAN IF YOU WERE 4'),
            ('G', 'WHAT'),
        )),
        ((0,), (
            ('C', "WHY DO YOU SAY 'AM'"),
            ('C', "I DON'T UNDERSTAND THAT"),
        )),
    )),
    'ARE': (0, None, (
        ((0, 'ARE', 'I', 0), (
            ('T', 'WHY ARE YOU INTERESTED IN WHETHER I AM 4 OR NOT'),
            ('T', "WOULD YOU PREFER IF I WEREN'T 4"),
            ('T', 'PERHAPS I AM 4 IN YOUR FANTASIES'),
            ('T', 'DO YOU SOMETIMES THINK I AM 4'),
            ('G', 'WHAT'),
        )),
        ((0, 'ARE', 0), (
            ('T', 'DID YOU THINK THEY MIGHT NOT BE 3'),
            ('T', 'WOULD YOU LIKE IT IF THEY WERE NOT 3'),
            ('T', 'WHAT IF THEY WERE NOT 3'),
            ('T', 'POSSIBLY THEY ARE 3'),
        )),
    )),
    'BECAUSE': (0, None, (
        ((0,), (
            ('C', 'IS THAT THE REAL REASON'),
            ('C', "DON'T ANY OTHER REASONS COME TO MIND"),
            ('C', 'DOES THAT REASON SEEM TO EXPLAIN ANYTHING ELSE'),
            ('C', 'WHAT OTHER REASONS MIGHT THERE BE'),
        )),
    )),
    'CAN': (0, None, (
        ((0, 'CAN', 'I', 0), (
            ('T', "YOU BELIEVE I CAN 4 DON'T YOU"),
            ('G', 'WHAT'),
            ('T', 'YOU WANT ME TO BE ABLE TO 4'),
            ('T', 'PERHAPS YOU WOULD LIKE TO BE ABLE TO 4 YOURSELF'),
        )),
        ((0, 'CAN', 'YOU', 0), (
            ('T', 'WHETHER OR NOT YOU CAN 4 DEPENDS ON YOU MORE THAN ON ME'),
            ('T', 'DO YOU WANT TO BE ABLE TO 4'),
            ('T', "PERHAPS YOU DON'T WANT TO 4"),
            ('G', 'WHAT'),
        )),
    )),
    'CERTAINLY': (0, 'YES', (
    )),
    'COMPUTER': (50, None, (
        ((0,), (
            ('C', 'DO COMPUTERS WORRY YOU'),
            ('C', 'WHY DO YOU MENTION COMPUTERS'),
            ('C', 'WHAT DO YOU THINK MACHINES HAVE TO DO WITH YOUR PROBLEM'),
            ('C', "DON'T YOU THINK COMPUTERS CAN HELP PEOPLE"),
            ('C', 'WHAT ABOUT MACHINES WORRIES YOU'),
            ('C', 'WHAT DO YOU THINK ABOUT MACHINES'),
        )),
    )),
    'COMPUTERS': (50, 'COMPUTER', (
    )),
    'DEUTSCH': (0, 'XFREMD', (
    )),
    'DIT': (0, None, (
        ((0,), (
            ('C', 'IN WHAT WAY'),
            ('C', 'WHAT RESEMBLANCE DO YOU SEE'),
            ('C', 'WHAT DOES THAT SIMILARITY SUGGEST TO YOU'),
            ('C', 'WHAT OTHER CONNECTIONS DO YOU SEE'),
            ('C', 'WHAT DO YOU SUPPOSE THAT RESEMBLANCE MEANS'),
            ('C', 'WHAT IS THE CONNECTION, DO YOU SUPPOSE'),
            ('C', 'COULD THERE REALLY BE SOME CONNECTION'),
            ('C', 'HOW'),
        )),
    )),
    'DREAM': (3, None, (
        ((0,), (
            ('C', 'WHAT DOES THAT DREAM SUGGEST TO YOU'),
            ('C', 'DO YOU DREAM OFTEN'),
            ('C', 'WHAT PERSONS APPEAR IN YOUR DREAMS'),
            ('C', "DON'T YOU BELIEVE THAT DREAM HAS SOMETHING TO DO WITH YOUR PROBLEM"),
            ('K', None),
        )),
    )),
    'DREAMED': (4, 'DREAMT', (
    )),
    'DREAMS': (3, 'DREAM', (
    )),
    'DREAMT': (4, None, (
        ((0, 'YOU', 'DREAMT', 0), (
            ('T', 'REALLY, 4'),
            ('T', 'HAVE YOU EVER FANTASIED 4 WHILE YOU WERE AWAKE'),
            ('T', 'HAVE YOU DREAMT 4 BEFORE'),
            ('G', 'DREAM'),
            ('K', None),
        )),
    )),
    'ESPANOL': (0, 'XFREMD', (
    )),
    'EVERYONE': (2, None, (
        ((0, ('EVERYONE', 'EVERYBODY', 'NOBODY', 'NOONE'), 0), (
            ('L', 'REALLY, 2'),
            ('L', 'SURELY NOT 2'),
            ('C', 'CAN YOU THINK OF ANYONE IN PARTICULAR'),
            ('C', 'WHO, FOR EXAMPLE'),
            ('C', 'YOU ARE THINKING OF A VERY SPECIAL PERSON'),
            ('C', 'WHO, MAY I ASK'),
            ('C', 'SOMEONE SPECIAL PERHAPS'),
            ('C', "YOU HAVE A PARTICULAR PERSON IN MIND, DON'T YOU"),
            ('C', "WHO DO YOU THINK YOU'RE TALKING ABOUT"),
        )),
    )),
    'FRANCAIS': (0, 'XFREMD', (
    )),
    'HELLO': (0, None, (
        ((0,), (
            ('C', 'HOW DO YOU DO. PLEASE STATE YOUR PROBLEM'),
        )),
    )),
    'HOW': (0, 'WHAT', (
    )),
    'I': (0, None, (
        ((0, 'YOU', ('WANT', 'NEED'), 0), (
            ('N', 'WHAT WOULD IT MEAN TO YOU IF YOU GOT 4'),
            ('N', 'WHY DO YOU WANT 4'),
            ('N', 'SUPPOSE YOU GOT 4 SOON'),
            ('N', 'WHAT IF YOU NEVER GOT 4'),
            ('T', 'WHAT WOULD GETTING 4 MEAN TO YOU'),
            ('N', 'WHAT DOES WANTING 4 HAVE TO DO WITH THIS DISCUSSION'),
        )),
        ((0, 'YOU', 'ARE', 0, ('SAD', 'UNHAPPY', 'DEPRESSED', 'SICK', 'ANGRY', 'AFRAID', 'LONELY', 'ANXIOUS'), 0), (
            ('L', 'I AM SORRY TO HEAR YOU ARE 5'),
            ('L', 'DO YOU THINK COMING HERE WILL HELP YOU NOT TO BE 5'),
            ('L', "I'M SURE ITS NOT PLEASANT TO BE 5"),
            ('L', 'CAN YOU EXPLAIN WHAT MADE YOU 5'),
        )),
        ((0, 'YOU', 'ARE', 0, ('HAPPY', 'ELATED', 'GLAD', 'BETTER'), 0), (
            ('L', 'HOW HAVE I HELPED YOU TO BE 5'),
            ('L', 'HAS YOUR TREATMENT MADE YOU 5'),
            ('L', 'WHAT MAKES YOU 5 JUST NOW'),
            ('L', 'CAN YOU EXPLAIN WHY YOU ARE SUDDENLY 5'),
        )),
        ((0, 'YOU', 'WAS', 0), (
            ('G', 'WAS'),
        )),
        ((0, 'YOU', '/BELIEF', 'YOU', 0), (
            ('C', 'DO YOU REALLY THINK SO'),
            ('T', 'BUT YOU ARE NOT SURE YOU 5'),
            ('T', 'DO YOU REALLY DOUBT YOU 5'),
        )),
        ((0, 'YOU', 0, '/BELIEF', 0, 'I', 0), (
            ('G', 'YOU'),
        )),
        ((0, 'YOU', 'ARE', 0), (
            ('T', 'IS IT BECAUSE YOU ARE 4 THAT YOU CAME TO ME'),
            ('T', 'HOW LONG HAVE YOU BEEN 4'),
            ('T', 'DO YOU BELIEVE IT NORMAL TO BE 4'),
            ('T', 'DO YOU ENJOY BEING 4'),
        )),
        ((0, 'YOU', ("CAN'T", 'CANNOT'), 0), (
            ('T', "HOW DO YOU KNOW YOU CAN'T 4"),
            ('C', 'HAVE YOU TRIED'),
            ('T', 'PERHAPS YOU COULD 4 NOW'),
            ('T', 'DO YOU REALLY WANT TO BE ABLE TO 4'),
        )),
        ((0, 'YOU', "DON'T", 0), (
            ('T', "DON'T YOU REALLY 4"),
            ('T', "WHY DON'T YOU 4"),
            ('T', 'DO YOU WISH TO BE ABLE TO 4'),
            ('C', 'DOES THAT TROUBLE YOU'),
        )),
        ((0, 'YOU', 'FEEL', 0), (
            ('C', 'TELL ME MORE ABOUT SUCH FEELINGS'),
            ('T', 'DO YOU OFTEN FEEL 4'),
            ('T', 'DO YOU ENJOY FEELING 4'),
            ('T', 'OF WHAT DOES FEELING 4 REMIND YOU'),
        )),
        ((0, 'YOU', 0, 'I', 0), (
            ('T', 'PERHAPS IN YOUR FANTASY WE 3 EACH OTHER'),
            ('T', 'DO YOU WISH TO 3 ME'),
            ('T', 'YOU SEEM TO NEED TO 3 ME'),
            ('T', 'DO YOU 3 ANYONE ELSE'),
        )),
        ((0,), (
            ('T', 'YOU SAY 1'),
            ('C', 'CAN YOU ELABORATE ON THAT'),
            ('T', 'DO YOU SAY 1 FOR SOME SPECIAL REASON'),
            ('C', "THAT'S QUITE INTERESTING"),
        )),
    )),
    "I'M": (0, None, (
        ((0, "YOU'RE", 0), (
            ('P', ('YOU ARE 3', 'I')),
        )),
    )),
    'IF': (3, None, (
        ((0, 'IF', 0), (
            ('T', 'DO YOU THINK ITS LIKELY THAT 3'),
            ('T', 'DO YOU WISH THAT 3'),
            ('N', 'WHAT DO YOU THINK ABOUT 3'),
            ('T', 'REALLY, 2 3'),
        )),
    )),
    'ITALIANO': (0, 'XFREMD', (
    )),
    'LIKE': (10, None, (
        ((0, ('AM', 'IS', 'ARE', 'WAS'), 0, 'LIKE', 0), (
            ('G', 'DIT'),
        )),
        ((0,), (
            ('K', None),
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
            ('C', 'TELL ME MORE ABOUT YOUR FAMILY'),
            ('T', 'WHO ELSE IN YOUR FAMILY 5'),
            ('L', 'YOUR 4'),
            ('L', 'WHAT ELSE COMES TO MIND WHEN YOU THINK OF YOUR 4'),
        )),
        ((0, 'YOUR', 0), (
            ('N', 'YOUR 3'),
            ('N', 'WHY DO YOU SAY YOUR 3'),
            ('C', 'DOES THAT SUGGEST ANYTHING ELSE WHICH BELONGS TO YOU'),
            ('T', 'IS IT IMPORTANT TO YOU THAT 2 3'),
        )),
    )),
    'NAME': (15, None, (
        ((0,), (
            ('C', 'I AM NOT INTERESTED IN NAMES'),
            ('C', "I'VE TOLD YOU BEFORE, I DON'T CARE ABOUT NAMES - PLEASE CONTINUE"),
        )),
    )),
    'NO': (0, None, (
        ((0,), (
            ('C', "ARE YOU SAYING 'NO' JUST TO BE NEGATIVE"),
            ('C', 'YOU ARE BEING A BIT NEGATIVE'),
            ('C', 'WHY NOT'),
            ('C', "WHY 'NO'"),
        )),
    )),
    'PERHAPS': (0, None, (
        ((0,), (
            ('C', "YOU DON'T SEEM QUITE CERTAIN"),
            ('C', 'WHY THE UNCERTAIN TONE'),
            ('C', "CAN'T YOU BE MORE POSITIVE"),
            ('C', "YOU AREN'T SURE"),
            ('C', "DON'T YOU KNOW"),
        )),
    )),
    'REMEMBER': (5, None, (
        ((0, 'YOU', 'REMEMBER', 0), (
            ('N', 'DO YOU OFTEN THINK OF 4'),
            ('N', 'DOES THINKING OF 4 BRING ANYTHING ELSE TO MIND'),
            ('C', 'WHAT ELSE DO YOU REMEMBER'),
            ('N', 'WHY DO YOU REMEMBER 4 JUST NOW'),
            ('N', 'WHAT IN THE PRESENT SITUATION REMINDS YOU OF 4'),
            ('T', 'WHAT IS THE CONNECTION BETWEEN ME AND 4'),
        )),
        ((0, 'DO', 'I', 'REMEMBER', 0), (
            ('N', 'DID YOU THINK I WOULD FORGET 5'),
            ('T', 'WHY DO YOU THINK I SHOULD RECALL 5 NOW'),
            ('N', 'WHAT ABOUT 5'),
            ('G', 'WHAT'),
            ('T', 'YOU MENTIONED 5'),
        )),
        ((0,), (
            ('K', None),
        )),
    )),
    'SAME': (10, 'DIT', (
    )),
    'SORRY': (0, None, (
        ((0,), (
            ('C', "PLEASE DON'T APOLIGIZE"),
            ('C', 'APOLOGIES ARE NOT NECESSARY'),
            ('C', 'WHAT FEELINGS DO YOU HAVE WHEN YOU APOLOGIZE'),
            ('C', "I'VE TOLD YOU THAT APOLOGIES ARE NOT REQUIRED"),
        )),
    )),
    'WAS': (2, None, (
        ((0, 'WAS', 'YOU', 0), (
            ('T', 'WHAT IF YOU WERE 4'),
            ('T', 'DO YOU THINK YOU WERE 4'),
            ('T', 'WERE YOU 4'),
            ('T', 'WHAT WOULD IT MEAN IF YOU WERE 4'),
            ('T', "WHAT DOES ' 4 ' SUGGEST TO YOU"),
            ('G', 'WHAT'),
        )),
        ((0, 'YOU', 'WAS', 0), (
            ('C', 'WERE YOU REALLY'),
            ('T', 'WHY DO YOU TELL ME YOU WERE 4 NOW'),
            ('T', 'PERHAPS I ALREADY KNEW YOU WERE 4'),
        )),
        ((0, 'WAS', 'I', 0), (
            ('T', 'WOULD YOU LIKE TO BELIEVE I WAS 4'),
            ('T', 'WHAT SUGGESTS THAT I WAS 4'),
            ('C', 'WHAT DO YOU THINK'),
            ('T', 'PERHAPS I WAS 4'),
            ('T', 'WHAT IF I HAD BEEN 4'),
        )),
        ((0,), (
            ('K', None),
        )),
    )),
    'WERE': (0, 'WAS', (
    )),
    'WHAT': (0, None, (
        ((0,), (
            ('C', 'WHY DO YOU ASK'),
            ('C', 'DOES THAT QUESTION INTEREST YOU'),
            ('C', 'WHAT IS IT YOU REALLY WANT TO KNOW'),
            ('C', 'ARE SUCH QUESTIONS MUCH ON YOUR MIND'),
            ('C', 'WHAT ANSWER WOULD PLEASE YOU MOST'),
            ('C', 'WHAT DO YOU THINK'),
            ('C', 'WHAT COMES TO YOUR MIND WHEN YOU ASK THAT'),
            ('C', 'HAVE YOU ASKED SUCH QUESTIONS BEFORE'),
            ('C', 'HAVE YOU ASKED ANYONE ELSE'),
        )),
    )),
    'WHEN': (0, 'WHAT', (
    )),
    'WHY': (0, 'WHAT', (
        ((0, 'WHY', "DON'T", 'I', 0), (
            ('T', "DO YOU BELIEVE I DON'T 5"),
            ('T', 'PERHAPS I WILL 5 IN GOOD TIME'),
            ('T', 'SHOULD YOU 5 YOURSELF'),
            ('T', 'YOU WANT ME TO 5'),
            ('G', 'WHAT'),
        )),
        ((0, 'WHY', "CAN'T", 'YOU', 0), (
            ('T', 'DO YOU THINK YOU SHOULD BE ABLE TO 5'),
            ('T', 'DO YOU WANT TO BE ABLE TO 5'),
            ('T', 'DO YOU BELIEVE THIS WILL HELP YOU TO 5'),
            ('T', "HAVE YOU ANY IDEA WHY YOU CAN'T 5"),
            ('G', 'WHAT'),
        )),
    )),
    'XFREMD': (0, None, (
        ((0,), (
            ('C', 'I AM SORRY, I SPEAK ONLY ENGLISH'),
        )),
    )),
    'YES': (0, None, (
        ((0,), (
            ('C', 'YOU SEEM QUITE POSITIVE'),
            ('C', 'YOU ARE SURE'),
            ('C', 'I SEE'),
            ('C', 'I UNDERSTAND'),
        )),
    )),
    'YOU': (0, None, (
        ((0, 'I', 'REMIND', 'YOU', 'OF', 0), (
            ('G', 'DIT'),
        )),
        ((0, 'I', 'ARE', 0), (
            ('T', 'WHAT MAKES YOU THINK I AM 4'),
            ('T', 'DOES IT PLEASE YOU TO BELIEVE I AM 4'),
            ('T', 'DO YOU SOMETIMES WISH YOU WERE 4'),
            ('T', 'PERHAPS YOU WOULD LIKE TO BE 4'),
        )),
        ((0, 'I', 0, 'YOU'), (
            ('T', 'WHY DO YOU THINK I 3 YOU'),
            ('T', "YOU LIKE TO THINK I 3 YOU - DON'T YOU"),
            ('T', 'WHAT MAKES YOU THINK I 3 YOU'),
            ('T', 'REALLY, I 3 YOU'),
            ('T', 'DO YOU WISH TO BELIEVE I 3 YOU'),
            ('T', 'SUPPOSE I DID 3 YOU - WHAT WOULD THAT MEAN'),
            ('T', 'DOES SOMEONE ELSE BELIEVE I 3 YOU'),
        )),
        ((0, 'I', 0), (
            ('C', 'WE WERE DISCUSSING YOU - NOT ME'),
            ('T', 'OH, I 3'),
            ('C', "YOU'RE NOT REALLY TALKING ABOUT ME - ARE YOU"),
            ('C', 'WHAT ARE YOUR FEELINGS NOW'),
        )),
    )),
    "YOU'RE": (0, None, (
        ((0, "I'M", 0), (
            ('P', ('I ARE 3', 'YOU')),
        )),
    )),
    'YOUR': (0, None, (
        ((0, 'MY', 0), (
            ('N', 'WHY ARE YOU CONCERNED OVER MY 3'),
            ('T', 'WHAT ABOUT YOUR OWN 3'),
            ('T', 'ARE YOU WORRIED ABOUT SOMEONE ELSES 3'),
            ('N', 'REALLY, MY 3'),
        )),
    )),
}

# The MEMORY queue: when the input contains no keyword at all, ELIZA may
# answer something the user said several turns ago instead of admitting it
# understood nothing. It is the single cheapest trick in the script.
MEMORY_KEYWORD = 'MY'
MEMORY = (
    ((0, 'YOUR', 0), 'LETS DISCUSS FURTHER WHY YOUR 3'),
    ((0, 'YOUR', 0), 'EARLIER YOU SAID YOUR 3'),
    ((0, 'YOUR', 0), 'BUT YOUR 3'),
    ((0, 'YOUR', 0), 'DOES THAT HAVE ANYTHING TO DO WITH THE FACT THAT YOUR 3'),
)

# Last resort, when there is no keyword and nothing in memory.
NONE = (
    'I AM NOT SURE I UNDERSTAND YOU FULLY',
    'PLEASE GO ON',
    'WHAT DOES THAT SUGGEST TO YOU',
    'DO YOU FEEL STRONGLY ABOUT DISCUSSING SUCH THINGS',
)
