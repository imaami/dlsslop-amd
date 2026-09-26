#!/usr/bin/bash
set -euo pipefail

readonly api="${GITHUB_API_URL:-https://api.github.com}"
readonly openai="${OPENAI_BASE_URL:-https://api.openai.com}"
readonly repo="${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is unset}"
readonly repo_owner="${GITHUB_REPOSITORY_OWNER:-${repo%%/*}}"
readonly event="${GITHUB_EVENT_PATH:?GITHUB_EVENT_PATH is unset}"
readonly prompt_file=".github/gpt-review/prompt.md"
readonly max_diff_bytes="${GPT_REVIEW_MAX_DIFF_BYTES:-600000}"
readonly max_bot_bytes="${GPT_REVIEW_MAX_BOT_BYTES:-120000}"
# The Responses API counts reasoning tokens against this cap along
# with the visible ones, and a reasoning model at xhigh spends tens
# of thousands before it writes a word: a cap sized for the review
# text alone ends the call before the review, as "incomplete" with
# no message at all.
readonly max_output_tokens="${GPT_REVIEW_MAX_OUTPUT_TOKENS:-64000}"
# The reviewer reads the repository itself, through one read-only
# git tool; each tool reply and the number of tool rounds are
# bounded so a curious model cannot flood its own context.
readonly max_tool_bytes="${GPT_REVIEW_MAX_TOOL_BYTES:-50000}"
readonly max_tool_calls="${GPT_REVIEW_MAX_TOOL_CALLS:-40}"
# The wall clock's share, measured from script entry: exploration
# rounds may spend this much, and everything after them lives on a
# fixed reserve -- ten minutes of forced finale, a comment post,
# the sweep -- so the worst case stays inside the job's mute
# twenty-five-minute timeout with the checkout's overhead counted.
# The finale is the largest call of the run (the whole transcript,
# xhigh reasoning, tool_choice none) and the Responses API sends
# nothing until the answer is complete: ninety seconds proved too
# tight once the PR grew, dying as timeouts with zero bytes
# received while every exploration round kept succeeding, and four
# minutes went the same way at thirty-four files -- the tool and
# transcript budgets spent in a round or two of parallel calls,
# the finale left to write the review of a seven-thousand-line
# diff in one call.
readonly max_seconds="${GPT_REVIEW_MAX_SECONDS:-600}"
readonly finale_seconds=600
readonly explore_by=$((SECONDS + max_seconds))
# The transcript is resent whole every round; forty rounds of tool
# replies can outgrow a context window mid-review, so crossing
# this forces the finale like the other budgets do.
readonly max_transcript_bytes="${GPT_REVIEW_MAX_TRANSCRIPT_BYTES:-786432}"

umask 077
tmp=$(mktemp -d)
reaction_id=
trap 'unreact_eyes || :; rm -rf "$tmp"' EXIT
mkdir "$tmp/home"

need()
{
	command -v "$1" >/dev/null || {
		printf 'gpt-review: missing command: %s\n' "$1" >&2
		exit 2
	}
}

need curl
need jq
need awk

[[ -r $prompt_file ]] || {
	printf 'gpt-review: cannot read %s\n' "$prompt_file" >&2
	exit 2
}

# A refused call says why in its body; that goes to the log, where
# the status alone said nothing about a token short of a permission.
gh_curl()
{
	local status
	curl --silent --show-error --fail-with-body \
		--connect-timeout 10 --max-time 120 \
		-H "Authorization: Bearer ${GH_TOKEN:?GH_TOKEN is unset}" \
		-H 'Accept: application/vnd.github+json' \
		-H 'X-GitHub-Api-Version: 2022-11-28' \
		"$@" > "$tmp/gh.out" || {
		status=$?
		printf 'gpt-review: GitHub API refused %s:\n' "${*: -1}" >&2
		head -c 2000 "$tmp/gh.out" >&2
		printf '\n' >&2
		return "$status"
	}
	cat "$tmp/gh.out"
}

# Every comment the job writes opens with a marker naming the PR and
# the head it looked at, so tooling can find the job's own comments
# among a PR's -- the failures too, which is how a reader learns why
# nothing came.  GitHub refuses a comment past 65536 characters; a
# review that long is cut and still posted.
post_comment()
{
	local body=$1

	if (( ${#body} > 65000 )); then
		body="${body:0:65000}"$'\n\n[... truncated by gpt-review ...]'
	fi
	body="<!-- gpt-review: pr=$pr sha=${head_sha:-} -->"$'\n'"$body"
	printf '%s' "$body" | jq -Rs '{body: .}' > "$tmp/comment.json"
	gh_curl \
		-X POST \
		-H 'Content-Type: application/json' \
		--data-binary "@$tmp/comment.json" \
		"$api/repos/$repo/issues/$pr/comments" >/dev/null
}

fail()
{
	post_comment "**GPT review failed:** $1"
	exit 1
}

# The eyes on the command are the first write the job makes, with
# the permission the review comment will need: a token short of it
# fails here, before the paid call, with the refusal in the log --
# nothing can reach the PR then, which is the point of knowing.
# The reaction's id is kept, and the exit trap takes the eyes off
# again whichever way the job ends: the other reviewers drop
# theirs when they finish, and lingering eyes read as work still
# under way.
react_eyes()
{
	printf '%s\n' '{"content":"eyes"}' > "$tmp/reaction.json"
	reaction_id=$(gh_curl \
		-X POST \
		-H 'Content-Type: application/json' \
		--data-binary "@$tmp/reaction.json" \
		"$api/repos/$repo/issues/comments/$comment_id/reactions" |
		jq -r '.id // empty') || {
		printf 'gpt-review: the token cannot write to the pull request; no review was asked\n' >&2
		exit 1
	}
}

unreact_eyes()
{
	[[ $reaction_id ]] || return 0
	gh_curl -X DELETE \
		"$api/repos/$repo/issues/comments/$comment_id/reactions/$reaction_id" \
		> /dev/null 2>&1 || :
}

# Cuts at a line boundary and then drains the rest: exiting early
# would close the pipe under the producer, and with pipefail a curl
# or jq killed by EPIPE fails the pipeline -- the whole review died
# on exactly the PRs large enough to need the cut.
truncate_lines()
{
	local limit=$1

	LC_ALL=C awk -v limit="$limit" '
		cut { next }
		{
			n += length($0) + 1
			if (n > limit) {
				print ""
				print "[... truncated by gpt-review ...]"
				cut = 1
				next
			}
			print
		}
	'
}

pr=$(jq -r '.issue.number // empty' "$event")
comment_id=$(jq -r '.comment.id // empty' "$event")
commenter=$(jq -r '.comment.user.login // empty' "$event")
comment_body=$(jq -r '.comment.body // ""' "$event")

[[ $pr =~ ^[0-9]+$ && $comment_id =~ ^[0-9]+$ ]] || {
	printf 'gpt-review: event is not a PR issue_comment\n' >&2
	exit 2
}

if [[ $commenter != "$repo_owner" || ${GITHUB_TRIGGERING_ACTOR:-} != "$repo_owner" ]]; then
	printf 'gpt-review: refusing command or rerun from %q (repository owner is %q)\n' \
		"${GITHUB_TRIGGERING_ACTOR:-$commenter}" "$repo_owner" >&2
	exit 0
fi

# The command stands at the start of a line, any line: one comment
# prods every bot, each on a line of its own.  The focus is what
# the comment says besides -- the rest of the first command line,
# kept whole even when it opens with / or @, and every other line
# that is not a command to some bot, which is to say not one that
# opens with / or @.  A note written for the reviewer reaches it,
# the other bots' orders do not, and each ORIGINAL line is judged
# exactly once: a filter run on stripped text mistook a focus
# opening with / for a bot command, and a substitution run on
# every line turned a second command line, or a /gpt reviewer,
# into focus.
# -c, not -q: -q hangs up at the first match, the writer upstream
# dies of SIGPIPE, and pipefail turned a valid command early in a
# long comment into "no command".  -c drains to the end.
printf '%s\n' "$comment_body" | tr -d '\r' |
	grep -c -E '^/gpt review( |$)' > /dev/null || {
	printf 'gpt-review: comment carries no /gpt review command line\n' >&2
	exit 0
}

focus=$(printf '%s\n' "$comment_body" | tr -d '\r' | LC_ALL=C awk '
	{
		if (!found && match($0, /^\/gpt review( +|$)/)) {
			found = 1
			$0 = substr($0, RLENGTH + 1)
		} else if ($0 ~ /^[\/@]/)
			next
		if ($0 == "") {
			if (out)
				gap = gap "\n"
			next
		}
		printf "%s%s\n", gap, $0
		gap = ""
		out = 1
	}
')

react_eyes

[[ -n ${OPENAI_API_KEY:-} ]] ||
	fail 'repository secret `OPENAI_API_KEY` is not configured.'

gh_curl "$api/repos/$repo/pulls/$pr" > "$tmp/pr.json" ||
	fail "could not read the pull request through the GitHub API."

title=$(jq -r '.title' "$tmp/pr.json")
pr_body=$(jq -r '.body // ""' "$tmp/pr.json")
author=$(jq -r '.user.login' "$tmp/pr.json")
base_ref=$(jq -r '.base.ref' "$tmp/pr.json")
head_ref=$(jq -r '.head.ref' "$tmp/pr.json")
head_sha=$(jq -r '.head.sha' "$tmp/pr.json")
changed=$(jq -r '.changed_files' "$tmp/pr.json")
additions=$(jq -r '.additions' "$tmp/pr.json")
deletions=$(jq -r '.deletions' "$tmp/pr.json")

# The PR head and base as refs in the trusted clone: git OBJECTS
# only, data the reviewer's read-only tool can browse.  Nothing of
# the PR is ever checked out, built, sourced or executed -- this
# job holds OPENAI_API_KEY -- and the working tree stays the
# default branch's.
auth=$(printf 'x-access-token:%s' "${GH_TOKEN:?}" | base64 -w0)
timeout 120 \
	git -c "http.${GITHUB_SERVER_URL:-https://github.com}/.extraheader=AUTHORIZATION: basic $auth" \
	fetch --quiet --no-tags "${GITHUB_SERVER_URL:-https://github.com}/$repo" \
	"pull/$pr/head:refs/gpt-review/head" \
	"+refs/heads/$base_ref:refs/gpt-review/base" ||
	fail "could not fetch the PR head and base as git refs."

# The reviewed commit is the one fetched, and everything hangs off
# it: the PR can move between the metadata read, the fetch and a
# diff served by REST, and then the label, the refs and the diff
# would name different commits.  So head_sha is re-read from the
# fetched ref and the orientation diff is cut locally from the
# pinned refs -- one source of truth, and no 300-file, 20000-line
# ceiling of the REST diff endpoint either.
head_sha=$(git rev-parse refs/gpt-review/head)
git --no-pager diff "refs/gpt-review/base...refs/gpt-review/head" |
	truncate_lines "$max_diff_bytes" > "$tmp/diff.txt" ||
	fail "could not cut the diff from the fetched refs."

# Existing AI-review feedback is useful mainly as a deduplication/adversarial
# signal. Restrict it to the reviewers on this repository -- CodeRabbit,
# Copilot, Codex -- and cap both per-comment and total size so a noisy
# review cannot consume the model context.  The
# feedback is context, not the subject: a page that will not come
# leaves the review to go without it.
for f in "issues.json:issues/$pr/comments" \
         "review-comments.json:pulls/$pr/comments" \
         "reviews.json:pulls/$pr/reviews"; do
	gh_curl "$api/repos/$repo/${f#*:}?per_page=100" > "$tmp/${f%%:*}" ||
		printf '[]' > "$tmp/${f%%:*}"
done

{
	jq -r '
		.[]
		| select((.user.login // "") | test("coderabbit|copilot|codex"; "i"))
		| "TOP-LEVEL [\(.user.login)]:\n\((.body // "")[0:5000])\n"
	' "$tmp/issues.json"

	jq -r '
		.[]
		| select((.user.login // "") | test("coderabbit|copilot|codex"; "i"))
		| "INLINE [\(.user.login)] \(.path // "?"):\(.line // .original_line // "?"):\n\((.body // "")[0:5000])\n"
	' "$tmp/review-comments.json"

	jq -r '
		.[]
		| select((.user.login // "") | test("coderabbit|copilot|codex"; "i"))
		| select((.body // "") != "")
		| "REVIEW [\(.user.login)] state=\(.state // "?"):\n\((.body // "")[0:5000])\n"
	' "$tmp/reviews.json"
} | truncate_lines "$max_bot_bytes" > "$tmp/bots.txt"

# The review policy comes from the trusted default branch. CLAUDE.md carries
# architecture and house rules; .coderabbit.yaml carries the concise
# reviewer-specific exceptions; AGENTS.md carries repository mechanics.
for f in CLAUDE.md .coderabbit.yaml AGENTS.md; do
	[[ -r $f ]] || continue
	{
		printf '\n===== %s =====\n' "$f"
		cat "$f"
	} >> "$tmp/policy.txt"
done
: > "${tmp}/empty"
[[ -e $tmp/policy.txt ]] || cp "$tmp/empty" "$tmp/policy.txt"

metadata=$(jq -n \
	--arg repo "$repo" \
	--argjson pr "$pr" \
	--arg title "$title" \
	--arg author "$author" \
	--arg base "$base_ref" \
	--arg head "$head_ref" \
	--arg sha "$head_sha" \
	--argjson changed "$changed" \
	--argjson additions "$additions" \
	--argjson deletions "$deletions" \
	'{repo:$repo, pr:$pr, title:$title, author:$author,
	  base:$base, head:$head, head_sha:$sha, changed_files:$changed,
	  additions:$additions, deletions:$deletions}')

# The reviewer's one tool: git, read-only, over the trusted clone
# and the fetched refs.  The subcommand whitelist keeps it off the
# network, and options are ALLOWLISTED by exact name: git accepts
# any unique abbreviation of a long option, so a denylist of full
# spellings is a sieve (--open-files-in-page= still reached grep's
# pager, and the pager is a command line).  A name not listed --
# abbreviated, misspelt or dangerous -- is refused, and the refusal
# is tool output the model reads and corrects course on.  Left out
# on purpose: everything that executes (-O and grep's pager,
# --ext-diff, --textconv, --filters), writes (--output and kin), or
# reads outside the object store (--no-index, blame --contents,
# --ignore-revs-file, --stdin, --batch).  Free arguments -- revs,
# paths, patterns -- pass; a path outside the repository is git's
# own error.  And should something slip anyway, it lands in an
# empty room: the tool runs under env -i, no OPENAI_API_KEY, no
# GH_TOKEN, no runner variables.
allowed_option()
{
	case ${1%%=*} in
	--|--abbrev|--abbrev-commit|--abbrev-ref|--after|--all|--all-match|\
	--ancestry-path|--and|--author|--author-date-order|--basic-regexp|\
	--before|--boundary|--branches|--break|--cached|--cherry|\
	--cherry-pick|--children|--color|--color-words|--column|--committer|\
	--contains|--count|--date|--date-order|--decorate|--diff-filter|\
	--do-walk|--email|--exclude|--exclude-standard|--extended-regexp|\
	--files-with-matches|--files-without-match|--find-copies|\
	--find-copies-harder|--find-renames|--first-parent|--fixed-strings|\
	--follow|--format|--full-diff|--full-history|--full-index|\
	--full-name|--full-tree|--function-context|--glob|--graph|--heading|\
	--heads|--ignore-all-space|--ignore-blank-lines|--ignore-space-at-eol|\
	--ignore-space-change|--incremental|--invert-grep|--left-only|\
	--left-right|--line-number|--line-porcelain|--long|--max-count|\
	--max-depth|--max-parents|--merge-base|--merged|--merges|\
	--min-parents|--name-only|--name-status|--no-abbrev|\
	--no-abbrev-commit|--no-color|--no-contains|--no-decorate|\
	--no-max-parents|--no-merged|--no-merges|--no-min-parents|\
	--no-patch|--no-walk|--not|--numbered|--numstat|--oneline|\
	--only-matching|--or|--others|--parents|--patch|--perl-regexp|\
	--pickaxe-all|--pickaxe-regex|--points-at|--porcelain|--pretty|\
	--raw|--regexp-ignore-case|--remotes|--reverse|--right-only|--root|\
	--shortstat|--show-email|--show-function|--show-name|--show-number|\
	--show-stats|--simplify-merges|--since|--skip|--sort|--source|\
	--sparse|--stage|--stat|--summary|--symbolic|--symbolic-full-name|\
	--short|--tags|--topo-order|--unified|--until|--verify|--word-diff|\
	--word-diff-regex)
		return 0 ;;
	esac
	case $1 in
	-[0-9]*|-p|-t|-s|-r|-l|-z|-m|-c|-h|-v|-i|-w|-b|-e*|-E|-F|-P|-W|\
	-n|-n[0-9]*|-U|-U[0-9]*|-A[0-9]*|-B[0-9]*|-C|-C[0-9]*|-M|-M[0-9]*|\
	-S*|-G*|-L*)
		return 0 ;;
	esac
	return 1
}

# A free argument is a rev, a pathspec or a pattern, and must not
# name the filesystem outside the repository: git diff handed two
# paths slides into no-index comparison without the option being
# spelled, and /etc/hostname would read straight past an option
# allowlist.  Absolute paths and .. path components are refused; a
# rev range keeps its dots, which sit between names, never between
# slashes.
free_arg_ok()
{
	case $1 in
	/*|..|../*|*/..|*/../*) return 1 ;;
	esac
	return 0
}

run_git()
{
	local sub=${1:-}
	case $sub in
	log|show|diff|grep|blame|ls-tree|ls-files|rev-list|rev-parse|shortlog|cat-file|describe) ;;
	*)
		printf 'gpt-review tool: git %q is not allowed; use: log show diff grep blame ls-tree ls-files rev-list rev-parse shortlog cat-file describe\n' "$sub"
		return 0
		;;
	esac
	shift
	local a
	for a in "$@"; do
		if [[ $a == -* ]]; then
			allowed_option "$a" || {
				printf 'gpt-review tool: option %q is not in the allowlist\n' "$a"
				return 0
			}
		else
			free_arg_ok "$a" || {
				printf 'gpt-review tool: %q leaves the repository\n' "$a"
				return 0
			}
		fi
	done
	# The model must see the seams: an unmarked prefix reads as
	# the whole file, and a swallowed timeout as an empty result.
	local status=0
	env -i PATH=/usr/bin:/bin HOME="$tmp/home" TZ=UTC \
		GIT_CONFIG_NOSYSTEM=1 GIT_TERMINAL_PROMPT=0 \
		timeout 30 git --no-pager "$sub" "$@" \
		> "$tmp/git.out" 2>&1 || status=$?
	head -c "$max_tool_bytes" "$tmp/git.out"
	if (( $(stat -c %s "$tmp/git.out") > max_tool_bytes )); then
		printf '\n[gpt-review: output truncated at %s bytes]\n' \
			"$max_tool_bytes"
	fi
	if (( status == 124 )); then
		printf '[gpt-review: git timed out after 30s]\n'
	elif (( status != 0 )); then
		printf '[gpt-review: git exited %d]\n' "$status"
	fi
}

jq -n \
	--rawfile instructions "$prompt_file" \
	--arg metadata "$metadata" \
	--arg pr_body "$pr_body" \
	--arg focus "$focus" \
	--rawfile policy "$tmp/policy.txt" \
	--rawfile bots "$tmp/bots.txt" \
	--rawfile diff "$tmp/diff.txt" \
	'[
		{
			role: "developer",
			content: [{type:"input_text", text:$instructions}]
		},
		{
			role: "user",
			content: [{
				type:"input_text",
				text: (
					"<PR_METADATA>\n" + $metadata + "\n</PR_METADATA>\n\n" +
					"<PR_DESCRIPTION>\n" + $pr_body + "\n</PR_DESCRIPTION>\n\n" +
					"<OPTIONAL_REVIEW_FOCUS>\n" + $focus + "\n</OPTIONAL_REVIEW_FOCUS>\n\n" +
					"<REPOSITORY_POLICY>\n" + $policy + "\n</REPOSITORY_POLICY>\n\n" +
					"<EXISTING_AI_REVIEWS>\n" + $bots + "\n</EXISTING_AI_REVIEWS>\n\n" +
					"<PR_DIFF>\n" + $diff + "\n</PR_DIFF>"
				)
			}]
		}
	]' > "$tmp/input.json"

# The tool loop, stateless on purpose (store: false): every round
# resends the whole transcript, reasoning items riding along
# encrypted, and each function call runs here and returns as a
# function_call_output item.  When the round budget is out the
# model is told to write with what it has.
tool_calls=0
git_calls=0
input_total=0
output_total=0
forced=
: > "$tmp/review.txt"
while :; do
	# One forced finale, whatever ran out -- rounds, clock or
	# context -- and the model is told which.  A finale that still
	# answers nothing fails aloud; the job timeout would have
	# killed everything in silence.
	# A round too close to the exploration deadline is not run a
	# little: it is the finale's turn, on the finale's own clock.
	# An exhausted remainder is never topped up.
	remaining=$((explore_by - SECONDS))
	spent=
	if (( tool_calls >= max_tool_calls )); then
		spent="the tool-call budget"
	elif (( remaining < 45 )); then
		spent="the clock"
	elif (( $(stat -c %s "$tmp/input.json") >= max_transcript_bytes )); then
		spent="the context budget"
	fi
	choice=auto
	if [[ $spent ]]; then
		[[ $forced ]] &&
			fail "out of time after $tool_calls tool call(s) with no review written."
		forced=1
		choice=none
		remaining=$finale_seconds
		jq --arg why "$spent" '. + [{role:"user", content:[{type:"input_text",
			text:("Spent: " + $why + ". Write the review from what you have seen.")}]}]' \
			"$tmp/input.json" > "$tmp/input.next" &&
			mv "$tmp/input.next" "$tmp/input.json"
	fi
	jq -n \
		--arg model "${OPENAI_MODEL:-gpt-6-astra}" \
		--arg effort "${OPENAI_REASONING_EFFORT:-xhigh}" \
		--argjson max_out "$max_output_tokens" \
		--arg choice "$choice" \
		--slurpfile input "$tmp/input.json" \
		'{
			model: $model,
			store: false,
			include: ["reasoning.encrypted_content"],
			reasoning: {effort: $effort},
			max_output_tokens: $max_out,
			tool_choice: $choice,
			tools: [{
				type: "function",
				name: "git",
				description: "Run one read-only git command in the repository clone. args is argv after git, e.g. [\"log\",\"--oneline\",\"-20\",\"refs/gpt-review/head\"]. Allowed subcommands: log show diff grep blame ls-tree ls-files rev-list rev-parse shortlog cat-file describe. Options are allowlisted by exact name, never abbreviated; a refused option names itself in the output. Revs, paths and patterns are free arguments.",
				strict: true,
				parameters: {
					type: "object",
					properties: {
						args: {type: "array", items: {type: "string"}}
					},
					required: ["args"],
					additionalProperties: false
				}
			}],
			input: $input[0]
		}' > "$tmp/request.json"

	# A transport failure leaves no HTTP code; curl's own exit is
	# the report then, and it reaches the PR like any other.  Each
	# round gets the loop's remaining time, never less than a
	# floor: a round that would cross the deadline is dead anyway,
	# and dying by curl posts its failure.
	http=$(curl --silent --show-error \
		--connect-timeout 10 --max-time "$remaining" \
		-o "$tmp/response.json" \
		-w '%{http_code}' \
		-X POST \
		-H "Authorization: Bearer $OPENAI_API_KEY" \
		-H 'Content-Type: application/json' \
		--data-binary "@$tmp/request.json" \
		"$openai/v1/responses") || http="curl exit $?"

	# Exit 28 is curl giving up on a response, not a network it
	# could not reach: the failure names the wait it made, so a
	# finale outgrown by the PR is not read as an outage.
	if [[ $http == "curl exit 28" ]]; then
		fail "OpenAI API sent no response within ${remaining}s ($http)."
	elif [[ ! $http =~ ^[0-9]+$ ]]; then
		fail "OpenAI API unreachable ($http)."
	elif [[ ! $http =~ ^2 ]]; then
		error=$(jq -r '.error.message // "unknown OpenAI API error"' \
			"$tmp/response.json" 2>/dev/null || printf 'unparseable OpenAI API error')
		fail "$(printf 'OpenAI API HTTP %s: %s' "$http" "$error")"
	fi

	input_total=$((input_total + $(jq -r '.usage.input_tokens // 0' "$tmp/response.json")))
	output_total=$((output_total + $(jq -r '.usage.output_tokens // 0' "$tmp/response.json")))

	# A round may speak and call tools in one breath; what it said
	# is kept as it comes, and the finale appends instead of
	# replacing -- only the last round's words reached the PR
	# before.
	text=$(jq -r '
		[
			.output[]?
			| select(.type == "message")
			| .content[]?
			| select(.type == "output_text")
			| .text
		] | join("\n")
	' "$tmp/response.json")
	if [[ $text ]]; then
		[[ -s $tmp/review.txt ]] && printf '\n\n' >> "$tmp/review.txt"
		printf '%s' "$text" >> "$tmp/review.txt"
	fi

	jq -s '.[0] + [.[1].output[]?]' "$tmp/input.json" "$tmp/response.json" \
		> "$tmp/input.next" && mv "$tmp/input.next" "$tmp/input.json"

	jq -c '.output[]? | select(.type == "function_call")' \
		"$tmp/response.json" > "$tmp/calls.jsonl"
	[[ -s $tmp/calls.jsonl ]] || break

	while IFS= read -r call; do
		call_id=$(jq -r '.call_id' <<< "$call")
		# A response may carry more calls than the budgets have
		# left -- rounds, the clock (forty serial calls of thirty
		# seconds each outrun the job timeout between two round
		# checks), or, with forty parallel calls of fifty
		# kilobytes each, the transcript itself; the surplus is
		# answered, never run, so the finale still happens and
		# its resend fits the window.
		if (( tool_calls >= max_tool_calls )); then
			printf 'gpt-review tool: budget spent\n' > "$tmp/tool.out"
		elif (( SECONDS + 30 >= explore_by )); then
			echo 'gpt-review tool: clock budget spent' > "$tmp/tool.out"
		elif (( $(stat -c %s "$tmp/input.json") >= max_transcript_bytes )); then
			printf 'gpt-review tool: context budget spent\n' > "$tmp/tool.out"
		else
			mapfile -t args < <(jq -r '(.arguments | fromjson).args[]?' <<< "$call")
			run_git "${args[@]}" > "$tmp/tool.out"
			((git_calls += 1))
		fi
		tool_calls=$((tool_calls + 1))
		jq --arg id "$call_id" --rawfile out "$tmp/tool.out" \
			'. + [{type:"function_call_output", call_id:$id, output:$out}]' \
			"$tmp/input.json" > "$tmp/input.next" &&
			mv "$tmp/input.next" "$tmp/input.json"
	done < "$tmp/calls.jsonl"
done

review=$(cat "$tmp/review.txt")

# A response the cap or the model cut short says so in its status;
# one with text is still posted, marked, and one without is a
# failure with its reason, not "no text output".
status=$(jq -r '.status // "completed"' "$tmp/response.json")
reason=$(jq -r '.incomplete_details.reason // .status // "unknown"' \
	"$tmp/response.json")

if [[ -z $review || $review == null ]]; then
	[[ $status == completed ]] ||
		fail "$(printf 'OpenAI response %s (%s) before any text was written.' \
			"$status" "$reason")"
	fail 'OpenAI returned no text output.'
fi

if [[ $status != completed ]]; then
	review+=$(printf '\n\n*Review cut short: response %s (%s).*' \
		"$status" "$reason")
fi

model=$(jq -r '.model // empty' "$tmp/response.json")
usage="$git_calls git call(s) · input $input_total, output $output_total tokens"

post_comment "$(cat <<EOF
## GPT review

Reviewed commit: \`$head_sha\`
$model · $usage

$review
EOF
)"
