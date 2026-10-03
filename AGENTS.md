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

## C

These apply to the project's C sources and to the headers that C and C++
share.

- Prefer C23 and later. The notion that some obsolete C standard represents
  "true C" is a golden age delusion. C is defined by the current standard.
- Use compile-time features such as `_Generic`, `typeof`, and `sizeof` to
  their full advantage.
- Use `int` only if needed. Habitual use of `int` introduces frequent integer
  conversions, which translates to costly sign extension instructions.
- Trace your call chains to see if you're e.g. calling `strlen()` multiple
  times over the same input. Measure once and pass down the variable.
- Stay aware of the program flow. Are you repeating some task more than once
  when you could just use a variable? Fix it.
- Use helper macros when it's justified and reasonable, but undefine macros
  that don't need to be exposed ASAP. Typically this means defining something
  above a function and undefining it below.
- If you call `strlen()` inside a loop condition or on a string literal, you
  must spend a day pushing a baby stroller full of boiled cabbage in public.
- Arrange struct members so that implicit padding is minimal. Wider types
  first, narrower towards the end, grouped by width.
- Be mindful of width guarantees. Given 3 struct members, an `int64_t`, a
  `long`, and an `int32_t`, `long` goes between the fixed-width types because
  its size could match either one. Remember the corner cases; e.g. `int` is
  only _required_ to be 16 bits.
- If a struct has trailing padding, and the last member is an integer type or
  `bool`, change the the type such that it occupies the padding, unless if it
  would introduce more complexity (such as additional casts downstream).
- Prefer RAII-like variable use. Initialize variables at declaration time when
  possible, but avoid initializing with a useless value out of habit.
- Don't declare variables at the top of a function without a reason. It's no
  longer idiomatic C. It's vestigial. Declare variables where they are needed,
  preferably when they can be RAII-initialized with a meaningful value.
- Scope variables as narrowly as possible.
- Use anonymous structs and unions to combat nesting hell as needed.
- Practically nothing should be a `typedef`. The situations where `typedef`
  is genuinely defensible are:
  - implementing opaque handle types in APIs (for example when instantiating
    a library returns an instance pointer);
  - API callback function types (note: _not_ callback _pointer_ types - more
    about this later);
  - uniform interface semantics for C and C++ users of an API
    (`typedef struct Foo Foo` lets C code pretend `struct` is implicit);
  - `unsigned _BitInt()` because aligning it vertically with much shorter
    type names is awful, and/or you need to type it often:
    ```c
    // this is kind of ok. regrettably.
    typedef unsigned _BitInt(48) u48_type;
    ```
- Structs and unions with a tag but no type alias are great despite being
  somewhat more verbose to type:
  - they make it possible to have RAII initializer functions that return by
    value and are named like the tag;
  - an alias can be a `struct`, `union`, or a number of other things, but a
    tag tells the type semantics immediately.
- Function type aliases should not be pointer type aliases. While the former
  has its use cases, the latter is
  - unnecessary because appending an asterisk makes it a function pointer
    just like with any other type alias;
  - harmful because all pointer type aliases obscure pointer-ness, which is
    fundamentally unacceptable;
  - limited because a function type can forward declare a function but a
    function pointer type can not.
- `PATH_MAX` is not for defining array sizes. Don't trust its value to be
  reasonable.
- Always initialize and invalidate file descriptors by setting them to `-1`.
- Don't use an external flag to track a file descriptor's state. `-1` means
  not in use. Always uphold this rule.
- Immediately set pointers to `nullptr` after calling `free()`. Passing a null
  pointer to `free()` is a no-op per the C standard.
- If you must compare a signed integer variable to a `sizeof` expression,
  don't cast the variable to `size_t`, cast `sizeof` to the signed type
  instead. The former is a runtime conversion but the latter is not.
- If you keep casting signed variables to unsigned or vice versa, you're not
  planning ahead. Think of what you need to accomplish and pick the types
  based on that.
- Always check the return value of libc calls when they're not completely
  inconsequential.
- Do not assign to `errno` as a way to pass error information forward.
- In some cases `errno` must be cleared of stale values to guarantee its
  usefulness, e.g. before calling `strtoul()` and related functions. These are
  documented workarounds for known gotchas, not a suggestion to adopt `errno`
  as your error propagation channel.

## C coding style

### Include statements

- Place angle-bracketed include statements above double-quoted ones. Separate
  these into groups by placing an empty line in between the two.
- A separating empty line is only mandatory between `<>` and `""` groups, but
  it is also allowed within these two groups at your discretion.
- Sort grouped includes - those not separated by empty lines - by header name
  in the C locale. Ignore whitespace so that e.g. `#include` and `# include`
  sort equal.

### Indentation and alignment

- Tabs indent, spaces align: continuation lines of a declaration or argument
  list are tab-indented to the statement's level, then padded with spaces into
  column alignment. Don't "fix" space-aligned continuations into tabs.
- Spotting trivial cases of tab abuse is easy - anything this matches is bad:
  ```bash
  git grep -P '[^\t]\t' -- '**\.[ch]'
  ```
  However that isn't the full story; line continuations need special handling.
- To understand the interaction between tab indentation, space alignment, and
  line continuations, you need to imagine switching tab width from 8 to 3. Do
  all continuation lines stay aligned? If not, think about why that is.

### Comments

- Use Doxygen comments `/** */` to document functions, structs, and unions.
  Comments inside function bodies must be ordinary non-doxygenated comments.

### Functions

- Function declarations and definitions resemble GNU style somewhat, save for
  for the GNU brace style and two-space indentation:
  ```c
  ret_type
  function_name (some_type  const **a,
                 other_type const  *b,
                 int                c)
  {
  	// ...
  }

  ```

### Headers

- Header guard macros are derived from project name + subdir + file name,
  end in a single underscore, and have a commented `#endif`:
  ```c
  #ifndef DLSSLOP_AMD_LAYER_FOOBAR_H_
  #define DLSSLOP_AMD_LAYER_FOOBAR_H_
  // ...
  #endif /* DLSSLOP_AMD_LAYER_FOOBAR_H_ */
  ```
