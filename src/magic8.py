"""Magic 8-Ball answers and rendering for the 200x200 e-paper panel.

Deliberately does no I/O beyond drawing into a framebuffer, so it can be
exercised without a display attached.
"""

import os

# The twenty canonical answers: ten affirmative, five non-committal, five
# negative. Keep all twenty and keep the ratio -- it is what makes the toy feel
# right.
ANSWERS = (
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
    "Reply hazy, try again",
    "Ask again later",
    "Better not tell you now",
    "Cannot predict now",
    "Concentrate and ask again",
    "Don't count on it",
    "My reply is no",
    "My sources say no",
    "Outlook not so good",
    "Very doubtful",
)

# framebuf's built-in font is 8x8, so a 200 px panel fits 25 characters -- but a
# full 25 would run edge to edge and collide with the border, so wrap at 23.
# Only "Concentrate and ask again" and "Better not tell you now" need two lines.
_COLS = 23
_CHAR_W = 8
_LINE_H = 12

BLACK = 0x00
WHITE = 0xFF


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


def wrap(text, cols=_COLS):
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


def _centered(fb, text, y, colour=BLACK):
    fb.text(text, (200 - len(text) * _CHAR_W) // 2, y, colour)


def draw_ball(fb, cx=100, cy=46, r=34):
    """The 8-ball: a filled black disc with a white '8' built from two rings.

    The 8x8 built-in font can't be scaled, so an '8' drawn with text() would be
    a speck inside a 68 px disc. Two stacked rings read correctly at this size.
    """
    fb.ellipse(cx, cy, r, r, BLACK, True)
    for ring_cy, ring_r in ((cy - 12, 11), (cy + 12, 14)):
        # Two passes for a 2 px stroke -- a 1 px white line on black is faint
        # after the panel's dithering.
        fb.ellipse(cx, ring_cy, ring_r, ring_r, WHITE, False)
        fb.ellipse(cx, ring_cy, ring_r - 1, ring_r - 1, WHITE, False)


def render(fb, answer, footer=None):
    """Draw a complete screen into the framebuffer. Does not refresh the panel."""
    fb.fill(WHITE)
    fb.rect(2, 2, 196, 196, BLACK)
    draw_ball(fb)
    fb.hline(14, 92, 172, BLACK)

    lines = wrap(answer)
    # Centre the block vertically in the space between the rule and the footer.
    top = 104 + max(0, (60 - len(lines) * _LINE_H) // 2)
    for index, line in enumerate(lines):
        _centered(fb, line, top + index * _LINE_H)

    if footer:
        _centered(fb, footer, 178)
