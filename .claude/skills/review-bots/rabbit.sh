#!/usr/bin/bash
# CodeRabbit PR plumbing -- deterministic only; judgment lives in
# SKILL.md, shared machinery in lib.sh.
# usage: rabbit.sh (fetch | reply <tid> <body> | resolve <tid> |
#                   unresolve <tid> | comment <body>)

BOT=coderabbitai
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
main "$@"
