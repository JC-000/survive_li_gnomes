#!/bin/sh
# Banned-content gate. Two protections, both hard failures:
#
#  1. PATTERNS (.banned-patterns): classes of personal identifier and
#     credential. Checked in text content and in file paths.
#  2. HUMAN VOICE: no audio file may enter git unless allowlisted below.
#     The user's voice recordings must never appear in a commit in any way --
#     they were once committed by accident and had to be expunged with a
#     history rewrite. gitignore is advisory (git add -f walks past it);
#     this is the enforcement.
#
#   check_banned.sh staged    # pre-commit: staged paths + added lines
#   check_banned.sh tree      # everything tracked at HEAD + worktree index
#   check_banned.sh history   # every blob and commit message ever
set -eu
cd "$(git rev-parse --show-toplevel)"
MODE="${1:-staged}"
PAT=.banned-patterns
# FAIL crosses pipeline subshells via a flag file -- `... | while read` runs
# the loop in a subshell where variable assignment is silently lost.
FLAG=/tmp/banned_fail.$$
rm -f "$FLAG"

# Audio: extensions that can carry a recorded voice. clips/laugh.raw is the
# one allowlisted audio file: a third-party sample pack clip, not the user.
AUDIO_RE='\.(wav|raw|pcm|pcmw|adp|mp3|flac|ogg|m4a|aif|aiff|pak)$'
AUDIO_ALLOW='^clips/laugh\.raw$'
# Directories that only ever hold the user's voice; nothing under them may
# be committed whatever its extension.
VOICE_DIRS='^(takes|takes-oov|enrol-takes|takes-[a-z-]+|recordings)(/|$)'

grep -v '^\s*#' "$PAT" | grep -v '^\s*$' > /tmp/banned.$$ || true
# The local overlay holds the actual sensitive literals (never committed;
# built by tools/setup_hooks.sh from the standing scrub-replay config).
[ -f .banned-patterns.local ] && grep -v '^\s*#' .banned-patterns.local >> /tmp/banned.$$ || true

# Identity: every commit must be authored and committed by the target
# noreply identity. A leaked personal email in the author field is the
# leak the scrub-replay project exists to clean up after; cheaper to refuse.
IDENT_OK='^JC-000 <3798556\+JC-000@users\.noreply\.github\.com>$'

check_paths() {
    while IFS= read -r f; do
        [ -z "$f" ] && continue
        if echo "$f" | grep -qE "$VOICE_DIRS"; then
            echo "BANNED: voice directory path staged/tracked: $f" >&2; touch "$FLAG"
        elif echo "$f" | grep -qE "$AUDIO_RE" && ! echo "$f" | grep -qE "$AUDIO_ALLOW"; then
            echo "BANNED: audio file not on the allowlist: $f" >&2; touch "$FLAG"
        fi
        if grep -qEf /tmp/banned.$$ <<PATHEOF
$f
PATHEOF
        then echo "BANNED: pattern match in path name: $f" >&2; touch "$FLAG"; fi
    done
}

case "$MODE" in
staged)
    git diff --cached --name-only --diff-filter=ACRM | check_paths
    if git diff --cached --diff-filter=ACM -U0 | grep '^+' | grep -v '^+++' | grep -nEf /tmp/banned.$$ >&2; then
        echo "BANNED: pattern match in staged content (lines above)" >&2; touch "$FLAG"
    fi
    ;;
tree)
    git ls-files | check_paths
    if git grep -nEf /tmp/banned.$$ -- . >&2; then
        echo "BANNED: pattern match in tracked content (above)" >&2; touch "$FLAG"
    fi
    ;;
history)
    if git log --all --format='%an <%ae>%n%cn <%ce>' | sort -u | grep -vE "$IDENT_OK" >&2; then
        echo "BANNED: non-target identity in author/committer fields (above)" >&2; touch "$FLAG"
    fi
    # every path ever
    git log --all --pretty=format: --name-only --diff-filter=A | sort -u | check_paths
    # every commit message ever
    if git log --all --pretty=%B | grep -nEf /tmp/banned.$$ >&2; then
        echo "BANNED: pattern match in a commit message (above)" >&2; touch "$FLAG"
    fi
    # every blob ever (content)
    for rev in $(git rev-list --all); do
        hits=$(git grep -lEf /tmp/banned.$$ "$rev" -- 2>/dev/null || true)
        if [ -n "$hits" ]; then
            printf '%s\n' "$hits" | sed "s/^/BANNED content @ /" >&2
            touch "$FLAG"
        fi
    done
    ;;
*) echo "usage: check_banned.sh staged|tree|history" >&2; exit 2 ;;
esac
rm -f /tmp/banned.$$
if [ -e "$FLAG" ]; then
    rm -f "$FLAG"
    echo "banned-content check ($MODE): FAILED (findings above)" >&2
    exit 1
fi
echo "banned-content check ($MODE): clean"
