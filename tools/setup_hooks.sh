#!/bin/sh
# Run once per clone. Installs the local (uncommitted) git hooks and builds
# the local banned-literal overlay from the standing scrub-replay config.
#
# Hooks are deliberately NOT committed: the scrub-replay policy strips
# .githooks/ from history (BANNED_PATHS), and the literal patterns must never
# live in a committed file -- a public banned-strings file that contains the
# strings defeats itself.
#
# NOTHING SENSITIVE IS EMBEDDED HERE. This script is committed, and a
# committed generator that contains the literals is the same self-defeat as a
# committed pattern file -- the gate itself caught an earlier version of this
# script doing exactly that (it blocked the commit that carried it). All
# literals are extracted from ~/Documents/scrub-replay-config.sh at run time.
set -eu
cd "$(git rev-parse --show-toplevel)"
mkdir -p .githooks
cat > .githooks/pre-commit <<'HOOK'
#!/bin/sh
exec "$(git rev-parse --show-toplevel)/tools/check_banned.sh" staged
HOOK
cat > .githooks/pre-push <<'HOOK'
#!/bin/sh
exec "$(git rev-parse --show-toplevel)/tools/check_banned.sh" tree
HOOK
chmod +x .githooks/pre-commit .githooks/pre-push
git config core.hooksPath .githooks

SCRUB="$HOME/Documents/scrub-replay-config.sh"
if [ -f "$SCRUB" ]; then
    {
        echo "# GENERATED from scrub-replay-config.sh -- never commit this file."
        # The banned-domains regex, one alternative per line. Bare-word
        # alternatives (no @ or dot -- the name fragments) get word
        # boundaries: the scrub tool redacts, where over-matching is safe,
        # but a COMMIT gate that blocks "adjusting" because it contains a
        # first name is unusable. [[:<:]]/[[:>:]] are BSD grep's word edges.
        sed -n 's/^BANNED_DOMAINS_REGEX="\(.*\)"$/\1/p' "$SCRUB" | tr '|' '\n' \
            | while IFS= read -r alt; do
                case "$alt" in
                *@*|*\\.*) printf '%s\n' "$alt" ;;
                *)         printf '[[:<:]]%s[[:>:]]\n' "$alt" ;;
                esac
            done
        # Every OLD_EMAILS entry, with two distinctions the flat list hides:
        #
        # - noreply addresses get a leading guard: the banned UNPREFIXED
        #   noreply is a substring of the ALLOWED prefixed identity
        #   (…+NAME@users.noreply…), so a bare pattern flags every legitimate
        #   author line and the gate's own identity check. `(^|[^+0-9])`
        #   matches the leak and not the substring-inside-allowed form.
        # - local-parts are emitted only for personal-domain addresses. A
        #   noreply's local-part is the public username, which legitimately
        #   appears everywhere; a personal local-part is the leak itself.
        #   A local-part WITH separators is distinctive alone (widened
        #   separators -- one is a substring of the English word
        #   "progress", so boundaries matter). A single plain word is only
        #   leak-shaped with an @ after it: one of them is also the
        #   ordinary English word "shadow", which this codebase uses in
        #   prose about symbol shadowing.
        sed -n '/^OLD_EMAILS=(/,/^)/p' "$SCRUB" \
            | grep -oE '"[^"]+@[^"]+"' | tr -d '"' \
            | while IFS= read -r em; do
                esc=$(printf '%s\n' "$em" | sed 's/[.@+]/\\&/g')
                case "$em" in
                *@users.noreply.github.com|noreply@*)
                    printf '(^|[^+0-9])%s\n' "$esc" ;;
                *)
                    printf '%s\n' "$esc"
                    local_part=$(printf '%s\n' "$em" | cut -d@ -f1)
                    case "$local_part" in
                    *[._+]*) printf '%s\n' "$local_part" | sed 's/[._+]/[._]/g' ;;
                    *)       printf '%s@\n' "$local_part" ;;
                    esac ;;
                esac
            done
    } | grep -v '^$' | sort -u > .banned-patterns.local
    n=$(grep -c . .banned-patterns.local || true)
    echo "local overlay written ($n patterns, extracted not embedded)"
else
    echo "warning: $SCRUB not found; only the committed class patterns apply" >&2
fi
echo "hooks active: $(git config core.hooksPath)"
