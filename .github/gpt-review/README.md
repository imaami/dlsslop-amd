# GPT PR review for dlsslop-amd

A deliberately small, comment-triggered OpenAI reviewer.

## What it does

On a pull request, a trusted maintainer can write the literal
command, one space, at the start of a line:

```text
/gpt review
```

or add a focus, on the command's line or in the lines around it:

```text
@coderabbitai full review
@codex review
/gpt review concurrency, ownership, UB, and error paths

Note to GPT: treat bloat as a flaw.
```

The command may stand on any line, so one comment can prod every bot.
Everything the comment says besides commands -- the rest of the
command's line and every line that does not open with `/` or `@` --
is sent to the model as the review focus; the other bots' orders are
not.

The workflow:

1. runs the trusted reviewer code from the repository's default branch;
2. reads PR metadata and the unified diff through the GitHub API;
3. reads existing CodeRabbit/Copilot/Codex review text as deduplication
   and disagreement context;
4. adds the repository's existing `CLAUDE.md`, `.coderabbit.yaml`, and
   `AGENTS.md` as house-policy context;
5. makes one OpenAI Responses API request;
6. posts the resulting review as a normal PR conversation comment,
   opening with `<!-- gpt-review: pr=N sha=H -->` so tooling can find
   the job's comments (failures carry the marker too).

It never starts Codex, never runs a persistent agent, and never executes
the PR branch.

## Install

The workflow and these files live on the default branch. The
`issue_comment` workflow must exist there before `/gpt review` can
trigger it. They came from the srtview repository and are kept in step
with it; `review.sh` and `sweep.sh` keep their `set -euo pipefail` from
there, an exception to the AGENTS.md shell rule for this CI plumbing.

## Configure

Create a repository Actions secret (Settings → Secrets and variables →
Actions) named:

```text
OPENAI_API_KEY
```

A repository secret rather than an environment secret on purpose: a
job that references an environment is a deployment, and GitHub pins
the latest one to the repository front page -- nonsense for a review
bot. What an environment bought (secrets only for jobs from the
default branch) matters little here: the job never checks out or
executes PR code, and `issue_comment` workflows always run from the
default branch anyway.

The API is billed separately from a ChatGPT Plus subscription.

Optional repository Actions variables:

```text
OPENAI_REVIEW_MODEL=gpt-6-astra
OPENAI_REVIEW_REASONING=xhigh
```

The script also honours `OPENAI_BASE_URL` (default
`https://api.openai.com`) and `GITHUB_API_URL`, which is how it is
exercised against local stand-ins for both services.

The workflow defaults to those values if the variables do not exist.

Those are also the workflow defaults. The model and reasoning variables remain
configurable so changing either does not require editing the workflow.

## Authorization

Only the repository owner's GitHub login can trigger a paid review:

```text
comment author == repository owner
workflow triggering actor == repository owner
```

Checking both values also prevents collaborators from spending credits by
rerunning an earlier owner-triggered workflow. Other users' `/gpt review`
comments do not start the job.

## Security model

This is an `issue_comment` workflow, so it runs from the default branch.
That is intentional.

The job has a repository secret. Therefore it must not check out, source,
build, test, or execute the PR head. The PR diff, PR text, and bot comments
are fetched only as untrusted data and passed to the model.

Do not "improve" this by checking out `${{ github.event.pull_request... }}`
or a `refs/pull/...` head and then running repository scripts.

## Files

- `.github/workflows/gpt-review.yml` — trigger and least-privilege token
  permissions (`contents: read`; `pull-requests: write`: the comment and
  the reaction are written to a pull request, which `issues: write` does
  not cover; `actions: write`: the sweep below).
- `.github/gpt-review/review.sh` — GitHub/OpenAI API plumbing.
- `.github/gpt-review/prompt.md` — review policy.
- `.github/gpt-review/sweep.sh` — deletes this workflow's skipped runs.
  Every comment on every PR wakes an `issue_comment` workflow and a
  skipped job still leaves a run in the Actions list; each real run
  sweeps them on its way out, and `gpt.sh sweep` does it by hand.

## Limits

The default caps are:

```text
GPT_REVIEW_MAX_DIFF_BYTES=600000
GPT_REVIEW_MAX_BOT_BYTES=120000
GPT_REVIEW_MAX_OUTPUT_TOKENS=64000
```

They can be set in the workflow environment if a future PR needs
different limits. Truncation happens on line boundaries, and each
existing-review surface is read one page (100 entries) deep. The output
cap counts the model's reasoning tokens as well as the review text, so
it is sized for `xhigh` reasoning, not for the few thousand tokens the
review itself takes; a response the cap cuts short is posted with a
note, or reported when it cut before any text.

The first version posts one top-level review comment rather than inline
threads. That keeps the machinery deterministic and avoids trusting an
LLM to manufacture GitHub diff positions. File/line references remain in
the review text.
