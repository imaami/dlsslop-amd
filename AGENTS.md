# Native Linux integration conventions

- Rename only project-introduced identifiers. Preserve upstream names and
  contracts even in patched source; establish their origin from the pinned
  originals.
- Command-line interfaces use getopt-style short and long options. Settings
  have direct options, such as `--working-scale 1`; do not add positional
  `set NAME VALUE` or `toggle NAME` command languages.
- Every option's default belongs in `--help`, including conditional defaults,
  required/unset values, environment-derived paths and disabled flags. Generate
  setting defaults from the same definitions used at initialization/reset.
- Project shell scripts and generated launchers use `#!/usr/bin/bash`, Bash
  conditionals and arrays. Do not use global `set -e`, `set -u` or equivalents.
  Handle consequential errors explicitly and preserve arguments and exit status.
- In C/C++, avoid `using namespace` and use `\n` rather than `std::endl`.
- Project C++ is built without exceptions or RTTI. Fallible functions return
  `dlsslop::Result` (`common/result.h`); only the vendor boundary files catch
  what vendored code throws. Choose implementations at compile time (CRTP,
  concepts, templates), not through virtual calls or type erasure.
- Update CLI examples and focused parsing/launcher checks when changing tools.
- Build with CMake and follow `VALIDATION.md` for automated and hardware tests.
