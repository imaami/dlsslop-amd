#!/usr/bin/bash
# GPT PR plumbing -- the comment-triggered reviewer under
# .github/gpt-review, posting as github-actions[bot]; deterministic
# only, judgment lives in SKILL.md, shared machinery in lib.sh.  It
# writes top-level PR comments, each opening with
# "<!-- gpt-review: pr=N sha=H -->", so it is found by that marker
# rather than by login; it has no review threads, so a finding is
# answered with a PR comment that links the review, and a review is
# resolved by minimizing it under GitHub's RESOLVED classifier.
# ask posts the /gpt review command, which spends API credits.
# shellcheck disable=SC2034,SC2154  # VERBS/USAGE read by lib.sh; ctx sets owner, name, pr
# usage: gpt.sh (fetch | ask [focus] | reply <id> <body> |
#                resolve <id> | unresolve <id> | comment <body>)

VERBS='ask comment fetch reply resolve sweep unresolve'
USAGE='fetch | ask [focus] | reply <id> <body> | resolve <id> | unresolve <id> | sweep | comment <body>'
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

# Every gpt-review comment on the PR, reviews and failures alike,
# in the inline-thread shape the skill reads: thread is the node
# id resolve/unresolve take, resolved its minimized state, sha the
# head the review looked at.  A failure has no model line, and
# capture() answers a miss with EMPTY, which try/catch never sees
# and which drops the whole row from the object constructor: the
# alternative operator is what turns the miss into "".  Every
# failure the workflow ever posted was invisible here before.
fetch()
{
	ctx
	gh api graphql --paginate \
		-f owner="$owner" -f repo="$name" -F pr="$pr" -f query='
query($owner:String!,$repo:String!,$pr:Int!,$endCursor:String){
  repository(owner:$owner,name:$repo){ pullRequest(number:$pr){
    comments(first:100,after:$endCursor){
      pageInfo{ hasNextPage endCursor }
      nodes{ id databaseId url body isMinimized minimizedReason
             author{login} }}}}}' \
		--jq '
.data.repository.pullRequest.comments.nodes[]
| select(.body | startswith("<!-- gpt-review:"))
| {source:"issue", thread:.id, id:.databaseId,
   resolved:.isMinimized, why:.minimizedReason, url,
   sha:(.body | capture("sha=(?<s>[0-9a-f]*)").s),
   failed:(.body | test("GPT review failed")),
   model:((.body
           | capture("(?m)^(?<m>[A-Za-z0-9._-]+) · ").m) // ""),
   body}' ||
		fail "gpt comments fetch (graphql)"
}

# The trigger.  The command may stand on any line; the reviewer
# reads every line that is not a command to some bot as its focus,
# the disclaimer included, which is harmless.
ask()
{
	[[ ${1:-} != *$'\n'* ]] || fail 'ask: the focus is one line'
	gh pr comment --body "/gpt review ${1:-}"$'\n\n'"$(disclaimer)" ||
		fail 'gpt ask'
}

# No thread to reply in: a PR comment that links the review it
# answers, so the audit trail reads both ways.
reply()
{
	[[ ${1:-} && ${2:-} ]] || fail "usage: reply <id> <body>"
	local url
	url=$(gh api graphql -f id="$1" -f query='
query($id:ID!){ node(id:$id){ ... on IssueComment { url }}}' \
		--jq .data.node.url) && [[ $url ]] ||
		fail "no gpt comment $1"
	gh pr comment --body "$(disclaimer)"$'\n\n'"In reply to $url:"$'\n\n'"$2" ||
		fail "reply to $1"
}

resolve()
{
	[[ ${1:-} ]] || fail "usage: resolve <id>"
	gh api graphql -f id="$1" -f query='
mutation($id:ID!){ minimizeComment(input:{subjectId:$id,classifier:RESOLVED})
{ minimizedComment { isMinimized }}}' \
		--jq .data.minimizeComment.minimizedComment.isMinimized ||
		fail "resolve $1"
}

# The Actions-list noise a comment-triggered workflow makes: every
# real run sweeps it on the way out, and this does it by hand with
# the caller's token, which needs actions: write on the repo.
sweep()
{
	ctx
	GH_TOKEN="${GH_TOKEN:-$(gh auth token)}" \
	GITHUB_REPOSITORY="$owner/$name" \
		bash "$(git rev-parse --show-toplevel)/.github/gpt-review/sweep.sh" ||
		fail 'sweep (does the token have actions: write?)'
}

unresolve()
{
	[[ ${1:-} ]] || fail "usage: unresolve <id>"
	gh api graphql -f id="$1" -f query='
mutation($id:ID!){ unminimizeComment(input:{subjectId:$id})
{ unminimizedComment { isMinimized }}}' \
		--jq .data.unminimizeComment.unminimizedComment.isMinimized ||
		fail "unresolve $1"
}

main "$@"
