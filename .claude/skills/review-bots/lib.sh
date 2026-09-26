# lib.sh -- shared PR plumbing for the review-bot entry points.
# For sourcing only: no shebang, no top-level statements, functions
# only.  The sourcing entry point sets BOT to the reviewer's login
# prefix, lowercase, then calls main; one with verbs or surfaces
# of its own defines those functions after sourcing and lists its
# verbs in VERBS.  Logins vary by surface --
# GraphQL says "coderabbitai" / "copilot-pull-request-reviewer",
# REST reviews append "[bot]", REST inline comments say "Copilot" --
# so every filter case-folds and prefix-matches.  gh api --jq is
# gojq with no --arg plumbing; the shell splices $BOT into the
# program text instead.

# shellcheck shell=bash

fail() { printf '%s: %s\n' "${0##*/}" "$*" >&2; exit 1; }

ctx()
{
	repo=$(gh repo view --json owner,name -q '.owner.login+" "+.name' 2>&1) ||
		fail "gh repo view: $repo"
	read -r owner name <<<"$repo"
	pr=$(gh pr view --json number -q .number 2>&1) ||
		fail "no PR for current branch: $pr"
}

disclaimer()
{
	local user
	user='@'$(gh api user -q '.login') || {
		echo "can't fetch gh account owner" >&2
		user='the owner of this GitHub account'
	}
	printf '(Comment written by %s with permission from %s)\n' \
	       "${CLAUDE_MODEL:-Claude Code}" "$user"
}

fetch()
{
	ctx

	# Buffer all three surfaces; emit nothing unless every fetch succeeded,
	# so a partial set can never masquerade as complete.
	inline=$(gh api graphql --paginate \
		-f owner="$owner" -f repo="$name" -F pr="$pr" -f query='
query($owner:String!,$repo:String!,$pr:Int!,$endCursor:String){
  repository(owner:$owner,name:$repo){ pullRequest(number:$pr){
    reviewThreads(first:100,after:$endCursor){
      pageInfo{ hasNextPage endCursor }
      nodes{ id isResolved isOutdated path line
        comments(first:50){ nodes{ author{login} body url }}}}}}}' \
		--jq '
.data.repository.pullRequest.reviewThreads.nodes[]
| select(.comments.nodes[0].author.login
         | ascii_downcase | startswith("'"$BOT"'"))
| {source:"inline", thread:.id, resolved:.isResolved,
   outdated:.isOutdated, path, line,
   url:.comments.nodes[0].url,
   replies:(.comments.nodes|length-1),
   body:.comments.nodes[0].body}') ||
		fail "inline fetch (graphql)"

	# CodeRabbit parks findings it cannot anchor to the diff in the
	# review body under "Outside diff range"; the flag keeps them
	# from hiding among boilerplate review shells.
	reviews=$(gh api "repos/$owner/$name/pulls/$pr/reviews" --paginate \
		--jq '.[]|select(.user.login
		                 | ascii_downcase | startswith("'"$BOT"'"))
		      |{source:"review",id,state,
		        outside:((.body // "")|contains("Outside diff range")),
		        body:(.body // "")}') ||
		fail "reviews fetch"

	issues=$(gh api "repos/$owner/$name/issues/$pr/comments" --paginate \
		--jq '.[]|select(.user.login
		                 | ascii_downcase | startswith("'"$BOT"'"))
		      |{source:"issue",id,url:.html_url,body}') ||
		fail "issue comments fetch"

	printf '%s\n%s\n%s\n' "$inline" "$reviews" "$issues"
}

reply()
{
	[[ $1 && $2 ]] || fail "usage: reply <thread-id> <body>"
	gh api graphql -f tid="$1" -f body="$(disclaimer)"$'\n\n'"$2" -f query='
mutation($tid:ID!,$body:String!){
  addPullRequestReviewThreadReply(
    input:{pullRequestReviewThreadId:$tid,body:$body})
  { comment { url } }}' \
		--jq .data.addPullRequestReviewThreadReply.comment.url ||
		fail "reply to $1"
}

setres()
{
	[[ $2 ]] || fail "usage: $1 <thread-id>"
	gh api graphql -f tid="$2" -f query="
mutation(\$tid:ID!){ ${1}ReviewThread(input:{threadId:\$tid})
{ thread { isResolved }}}" \
		--jq ".data.${1}ReviewThread.thread.isResolved" ||
		fail "$1 $2"
}

resolve() { setres resolve "$1"; }
unresolve() { setres unresolve "$1"; }

comment()
{
	[[ $1 ]] || fail 'usage: comment <body>'
	gh pr comment --body "$(disclaimer)"$'\n\n'"$1" || fail "pr comment"
}

main()
{
	local verb
	for verb in ${VERBS:-comment fetch reply resolve unresolve}; do
		[[ ${1:-} == "$verb" ]] && { "$@"; return; }
	done
	fail "usage: ${USAGE:-fetch | reply <tid> <body> | resolve <tid> | unresolve <tid> | comment <body>}"
}
