"""Rendering a whole sentence on the 200x200 panel, for the ELIZA program.

`magic8.render` draws one short answer under an 8-ball. A DOCTOR reply is a
sentence, and dropping the graphic frees the whole panel for it.

Like magic8, this does no I/O beyond drawing into a framebuffer, so it can be
exercised without a display attached.

## The numbers

framebuf's built-in font is 8x8 and cannot be scaled, so text is pixel-doubled
(magic8._text_scaled). At scale 2 a glyph is 16x16:

| | scale 2 | scale 1 |
| --- | --- | --- |
| columns | 12 (192 px of 200) | 23 |
| line height | 18 px (16 + 2 leading) | 12 px |
| lines in the text area | 9 | 13 |
| character cells | 108 | 299 |

The research note behind this program quotes "12 cols x 15 lines at scale 2".
The 12 is right; the 15 is not possible -- 15 lines of 16 px glyphs is 240 px on
a 200 px panel. 9 lines is what fits above a footer.

9 is also, measured, exactly enough and no more. All 191 reply templates in
`eliza_rules` plus its NONE and MEMORY lines -- 199 in total -- with a six-word
echo substituted into *every* slot and put through `fit()`: worst case **9 lines
at scale 2**, longest 95 characters ("IS IT IMPORTANT TO YOU THAT 2 3", which
has two slots), and **no template falls back to scale 1**. So the box is full at
the top end. Measured independently on the ELIZA side at 9 lines too.

(191 is reply templates: the C/L/N/T forms. 211 counts the G/K/P control forms,
which never reach the panel, and 219 counts NONE and MEMORY as well. An earlier
figure of 202 here was a bad count -- it walked every string in the rule
structure rather than counting by template type.)
If the echo vocabulary ever gains longer words, this is the number that gives
first, and it gives by dropping a reply to scale 1 rather than by truncating --
`fit()` only truncates past 13 lines at scale 1, which nothing can reach.

Text is left-aligned, unlike magic8's centred one-liners. Centred prose reads as
a poem.
"""

import magic8

BLACK = magic8.BLACK
WHITE = magic8.WHITE

_CHAR_W = 8

# (scale, columns, max_lines), largest first. See the table above.
_SIZES = ((2, 12, 9), (1, 23, 13))

_TEXT_LEFT = 4
_TEXT_TOP = 10
_FOOTER_Y = 180


def _line_height(scale):
    return scale * _CHAR_W + (2 if scale == 2 else 4)


def fit(text):
    """Largest (scale, lines) this reply fits in, without splitting a word.

    Same rule as magic8.fit and for the same reason: a word broken across lines
    reads worse than one size smaller. The difference is that this has no
    3x size -- a sentence at 3x is eight columns wide and unreadable as prose.
    """
    longest_word = max(len(word) for word in text.split()) if text.split() else 1
    for scale, cols, max_lines in _SIZES:
        if longest_word > cols:
            continue
        lines = magic8.wrap(text, cols)
        if len(lines) <= max_lines:
            return scale, lines
    # Nothing fits: 1x and truncated. Losing the tail of a sentence is bad, but
    # scribbling over the footer is worse, and at 299 character cells this is
    # unreachable for anything DOCTOR can produce.
    scale, cols, max_lines = _SIZES[-1]
    return scale, magic8.wrap(text, cols)[:max_lines]


def render(fb, reply, footer=None):
    """Draw a complete reply screen. Does not refresh the panel."""
    fb.fill(WHITE)
    fb.rect(2, 2, 196, 196, BLACK)

    scale, lines = fit(reply)
    line_h = _line_height(scale)
    for index, line in enumerate(lines):
        # _text_scaled is magic8's, deliberately reused rather than copied: it is
        # pure framebuf, it is the only scaled-text renderer in the project, and
        # two copies would drift.
        magic8._text_scaled(fb, line, _TEXT_LEFT, _TEXT_TOP + index * line_h, BLACK, scale)

    if footer:
        fb.hline(6, _FOOTER_Y - 6, 188, BLACK)
        magic8._text_scaled(fb, footer[:24], _TEXT_LEFT, _FOOTER_Y, BLACK, 1)
