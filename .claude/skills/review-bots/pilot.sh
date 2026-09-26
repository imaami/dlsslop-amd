#!/usr/bin/bash
# Copilot PR plumbing -- deterministic only; judgment lives in
# SKILL.md, shared machinery in lib.sh.
# usage: pilot.sh (fetch | reply <tid> <body> | resolve <tid> |
#                  unresolve <tid> | comment <body>)

BOT=copilot
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
main "$@"
