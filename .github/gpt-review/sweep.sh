#!/usr/bin/bash
# Deletes this workflow's skipped runs.  Every comment on every PR
# wakes an issue_comment workflow; the job's gate skips the ones
# that are not commands, and each skip still leaves a run in the
# Actions list forever.  The runs are noise by construction --
# nothing executed -- so the real runs sweep them: the review job
# ends with this, whatever became of the review, and the
# review-bots tooling can run it by hand.  Only skipped runs of
# this one workflow; failures, cancellations and successes are
# history and stay.
set -euo pipefail

readonly api="${GITHUB_API_URL:-https://api.github.com}"
readonly repo="${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is unset}"
readonly workflow="${GPT_REVIEW_WORKFLOW:-gpt-review.yml}"

gh_curl()
{
	curl --silent --show-error --fail-with-body \
		--connect-timeout 10 --max-time 60 \
		-H "Authorization: Bearer ${GH_TOKEN:?GH_TOKEN is unset}" \
		-H 'Accept: application/vnd.github+json' \
		-H 'X-GitHub-Api-Version: 2022-11-28' \
		"$@"
}

# The HTTP status alone, for the one call whose failure modes must
# be told apart instead of tolerated wholesale.
gh_status()
{
	curl --silent -o /dev/null -w '%{http_code}' \
		--connect-timeout 10 --max-time 60 \
		-H "Authorization: Bearer ${GH_TOKEN:?GH_TOKEN is unset}" \
		-H 'Accept: application/vnd.github+json' \
		-H 'X-GitHub-Api-Version: 2022-11-28' \
		"$@" || printf 'curl exit %d' "$?"
}

# Page 1 again and again until it comes back empty: deleting from
# the page walked would skip every other entry, and past a hundred
# skips one page was all that got swept.  Sweeps may overlap, so a
# 404 is a rival's success, noted and skipped -- but ONLY a 404: a
# 403 is a token short of actions: write, a 5xx is GitHub's hour,
# and calling either "already gone" left every skipped run behind
# with a green exit.  Any other status is reported and fails the
# sweep; a pass that deletes nothing stops the loop rather than
# spinning on it.
swept=0
failed=0
last_first=
while :; do
	pass=0
	gone=0
	ids=$(gh_curl "$api/repos/$repo/actions/workflows/$workflow/runs?status=skipped&per_page=100" |
		jq -r '.workflow_runs[].id')
	[[ $ids ]] || break
	first=${ids%%$'\n'*}
	for id in $ids; do
		status=$(gh_status -X DELETE "$api/repos/$repo/actions/runs/$id")
		case $status in
		204)
			printf 'gpt-review: swept skipped run %s\n' "$id"
			pass=$((pass + 1))
			;;
		404|410)
			printf 'gpt-review: run %s already swept by a rival\n' "$id"
			gone=$((gone + 1))
			;;
		*)
			printf 'gpt-review: run %s not deleted: %s\n' \
				"$id" "$status" >&2
			failed=1
			;;
		esac
	done
	swept=$((swept + pass))
	# A page the rival emptied is progress too -- the state moved,
	# and more runs may have shifted onto page one -- but the SAME
	# page coming back with nothing deleted is the listing lagging
	# behind, not work remaining: without that guard two lagging
	# sweeps would spin on each other's ghosts.
	if (( pass == 0 )); then
		[[ $first != "$last_first" ]] && (( gone > 0 )) || break
	fi
	last_first=$first
done
printf 'gpt-review: %d skipped run(s) swept\n' "$swept"
exit "$failed"
