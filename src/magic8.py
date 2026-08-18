"""Magic 8-Ball answers and rendering for the 200x200 e-paper panel.

Deliberately does no I/O beyond drawing into a framebuffer, so it can be
exercised without a display attached.
"""

import os

import framebuf

# Answers are grouped by disposition, because the ratio between the groups is
# what gives the toy its character -- a ball that mostly says no feels mean
# rather than playful. Weizenbaum's-era 8-Balls were deliberately 50% yes.
#
# Current mix: 18 affirmative / 12 non-committal / 14 negative = 41 / 27 / 32 %.
# The classic twenty on their own were 50 / 25 / 25. Adding the darker set
# shifted it; if it ever starts feeling sour, that is the number to look at.
ANSWERS = (
    # --- the canonical twenty ---------------------------------------------
    # affirmative
    "It is certain",
    "It is decidedly so",
    "Without a doubt",
    "Yes definitely",
    "You may rely on it",
    "As I see it, yes",
    "Most likely",
    "Outlook good",
    "Yes",
    "Signs point to yes",
    # non-committal
    "Reply hazy, try again",
    "Ask again later",
    "Better not tell you now",
    "Cannot predict now",
    "Concentrate and ask again",
    # negative
    "Don't count on it",
    "My reply is no",
    "My sources say no",
    "Outlook not so good",
    "Very doubtful",

    # --- darker additions --------------------------------------------------
    # affirmative
    "Yes, for now",
    "The bones say yes",
    "Yes, but you'll regret it",
    "Yes, and no one will know",
    "Inevitably, yes",
    "Yes, tragically",
    "Yes. Consequences later",
    "It is written. Sorry",
    # non-committal
    "Better you don't know",
    "Ask me when it matters",
    "The void declines to comment",
    "Concentrate on your affairs",
    "Outlook obscured by dread",
    "That is between you and time",
    "Reply hazy. Like your future",
    # negative
    "Outlook fatal",
    "No, but nice try",
    "Not in this lifetime",
    "Signs point to therapy",
    "No, and stop asking",
    "Ask your next of kin",
    "My sources have gone quiet",
    "Doubtful. Look behind you",
    "Don't count on anything",
)

BLACK = 0x00
WHITE = 0xFF

_CHAR_W = 8  # framebuf's built-in font; not scalable, hence _text_scaled below

# Vertical layout. The ball is deliberately smaller than it could be: legible
# text matters more than a big graphic.
_BALL_CY = 40
_BALL_R = 28
_RULE_Y = 78
_ANSWER_TOP = 88
_ANSWER_H = 80
_FOOTER_Y = 180

# Candidate text sizes, largest first: (scale, columns, max_lines).
# Columns are 200 // (8 * scale) minus a margin; max_lines is _ANSWER_H // line
# height. render() picks the first size the answer fits in without splitting a
# word mid-way.
_SIZES = ((3, 8, 2), (2, 12, 4), (1, 23, 6))


def _line_height(scale):
    return scale * _CHAR_W + 4


def _rand_below(n):
    """Uniform random int in [0, n).

    os.urandom is backed by the RP2350's hardware RNG. Rejection sampling keeps
    it uniform -- plain modulo would bias the low answers.
    """
    limit = 256 - (256 % n)
    while True:
        value = os.urandom(1)[0]
        if value < limit:
            return value % n


def pick(exclude=None):
    """A random answer, never the same one twice in a row."""
    if len(ANSWERS) < 2:
        return ANSWERS[0]
    while True:
        answer = ANSWERS[_rand_below(len(ANSWERS))]
        if answer != exclude:
            return answer


def wrap(text, cols):
    """Greedy word wrap. Words longer than a line are hard-split."""
    lines = []
    line = ""
    for word in text.split():
        while len(word) > cols:
            if line:
                lines.append(line)
                line = ""
            lines.append(word[:cols])
            word = word[cols:]
        if not line:
            line = word
        elif len(line) + 1 + len(word) <= cols:
            line += " " + word
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def fit(text):
    """Largest (scale, lines) this text fits in the answer area.

    Rejects a size if any single word would have to be hard-split, since a word
    broken across lines reads far worse than one size smaller.
    """
    longest_word = max(len(w) for w in text.split())
    for scale, cols, max_lines in _SIZES:
        if longest_word > cols:
            continue
        lines = wrap(text, cols)
        if len(lines) <= max_lines:
            return scale, lines
    return 1, wrap(text, _SIZES[-1][1])


def _text_scaled(fb, text, x, y, colour, scale):
    """Draw text at an integer scale by pixel-doubling the 8x8 built-in font.

    framebuf has no scaled blit and the built-in font is fixed at 8x8, so the
    glyphs are rendered into a scratch mono buffer and each set pixel is painted
    as a scale x scale block. Costs a few tens of ms for a screen of text, which
    is nothing against a ~2.6 s panel refresh.
    """
    if scale == 1:
        fb.text(text, x, y, colour)
        return
    width = len(text) * _CHAR_W
    stride = (width + 7) // 8
    glyphs = framebuf.FrameBuffer(bytearray(stride * 8), width, 8, framebuf.MONO_HLSB)
    glyphs.fill(0)
    glyphs.text(text, 0, 0, 1)
    for row in range(8):
        for col in range(width):
            if glyphs.pixel(col, row):
                fb.fill_rect(x + col * scale, y + row * scale, scale, scale, colour)


def _centered(fb, text, y, scale=1, colour=BLACK):
    x = (200 - len(text) * _CHAR_W * scale) // 2
    _text_scaled(fb, text, x, y, colour, scale)


def draw_ball(fb, cx=100, cy=_BALL_CY, r=_BALL_R):
    """The 8-ball: a filled black disc with a white '8' built from two rings.

    The 8x8 built-in font can't be scaled by framebuf, so an '8' drawn with
    text() would be a speck inside the disc. Two stacked rings read correctly.
    """
    fb.ellipse(cx, cy, r, r, BLACK, True)
    for ring_cy, ring_r in ((cy - 10, 9), (cy + 10, 12)):
        # Two passes for a 2 px stroke -- a 1 px white line on black is faint
        # after the panel's dithering.
        fb.ellipse(cx, ring_cy, ring_r, ring_r, WHITE, False)
        fb.ellipse(cx, ring_cy, ring_r - 1, ring_r - 1, WHITE, False)


def render(fb, answer, footer=None):
    """Draw a complete screen into the framebuffer. Does not refresh the panel."""
    fb.fill(WHITE)
    fb.rect(2, 2, 196, 196, BLACK)
    draw_ball(fb)
    fb.hline(14, _RULE_Y, 172, BLACK)

    scale, lines = fit(answer)
    line_h = _line_height(scale)
    top = _ANSWER_TOP + max(0, (_ANSWER_H - len(lines) * line_h) // 2)
    for index, line in enumerate(lines):
        _centered(fb, line, top + index * line_h, scale)

    if footer:
        _centered(fb, footer, _FOOTER_Y)
