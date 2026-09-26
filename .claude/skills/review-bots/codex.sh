#!/usr/bin/bash
# Codex PR plumbing -- deterministic only; judgment lives in
# SKILL.md, shared machinery in lib.sh.  The GitHub login is
# chatgpt-codex-connector; the prefix below matches it.
# usage: codex.sh (fetch | reply <tid> <body> | resolve <tid> |
#                  unresolve <tid> | comment <body>)

BOT=chatgpt-codex
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
main "$@"
