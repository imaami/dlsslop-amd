#!/usr/bin/bash
# Runs dlsslopd once under the tracing HIP runtime; VALIDATION.md describes the
# procedure.

usage_line='Usage: trace.sh [OPTIONS]'
self=$(readlink -f -- "${BASH_SOURCE[0]}")
root=${self%/tests/hiptrace/*}
build_default=$root/build
build=$build_default
daemon=
modules_default=$root/assets/HIP/gfx1201
modules=$modules_default
assets_default=${XDG_DATA_HOME:-$HOME/.local/share}/dlsslop-amd/model
assets=$assets_default
output_default=$PWD
output=$output_default
channel_default=${XDG_RUNTIME_DIR:+$XDG_RUNTIME_DIR/dlsslop-hiptrace}
channel_default=${channel_default:-/tmp/dlsslop-hiptrace-$UID}
channel=$channel_default
name=
tier_default=720
tier=$tier_default
self_test=
performance=
passes_default=1
passes=$passes_default
motion=
frames_default=AAB
frames=
deep=
deep_kernels=
declare -A option_variable=(
    [-b]=build [--build]=build [-d]=daemon [--daemon]=daemon [-m]=modules [--modules]=modules
    [-a]=assets [--assets]=assets [-o]=output [--output]=output [-c]=channel [--channel]=channel
    [-n]=name [--name]=name [-t]=tier [--tier]=tier [-P]=passes [--passes]=passes
    [-F]=frames [--frames]=frames [-D]=deep [--deep]=deep [-k]=deep_kernels [--deep-kernels]=deep_kernels)

fail() {
    printf '%s\n' "$2" >&2
    exit "$1"
}

usage() {
    cat <<HELP
${usage_line}
Run dlsslopd once with the tracing HIP runtime from the build tree, which logs
every HIP call to OUTPUT/NAME.trace. dlsslopd reads no settings file and uses
its own channel in --channel. Served frames come from shmclient: its
output goes to OUTPUT/NAME.log and the daemon's to OUTPUT/NAME.daemon.log. A
self-test's output goes to OUTPUT/NAME.log. The client's or the self-test's
output is also shown; the daemon's log is only written to the file. The exit
status is the client's or the self-test's; 2 is a usage error.

Options:
  -b, --build DIR         Build tree with libhiptrace.so and shmclient
                          (default: ${build_default}).
  -d, --daemon FILE       The dlsslopd to trace (default: BUILD/dlsslopd).
  -m, --modules DIR       HIP modules (.hsaco) for dlsslopd, which also name the
                          modules in the trace
                          (default: ${modules_default}).
  -a, --assets DIR        HIP model weights
                          (default: ${assets_default}:
                          XDG_DATA_HOME/dlsslop-amd/model when XDG_DATA_HOME is
                          set and nonempty, otherwise ~/.local/share/dlsslop-amd/model).
  -o, --output DIR        Where the trace and logs go; created if missing
                          (default: the working directory,
                          ${output_default}).
  -c, --channel DIR       Private directory for the channel file NAME.bin;
                          created with mode 0700 if missing
                          (default: ${channel_default}:
                          XDG_RUNTIME_DIR/dlsslop-hiptrace when XDG_RUNTIME_DIR
                          is set and nonempty, otherwise /tmp/dlsslop-hiptrace-\$UID).
  -n, --name NAME         The trace's and logs' name (default: the configuration:
                          serve or selftest, the tier, then -perf, -motion and
                          -passesN when given, such as serve-720-perf).
  -t, --tier HEIGHT       Neural raster: 720, 900 or 1080; frames are served at
                          1280x720, 1600x900 or 1920x1080 (default: ${tier_default}).
  -S, --self-test         Run dlsslopd --self-test instead of serving frames
                          (default: off).
  -p, --performance       Pass --performance to dlsslopd (default: off).
  -P, --passes N          Chained evaluations per frame (default: ${passes_default}).
  -M, --motion            Serve with motion history (default: off).
  -F, --frames SPEC       Served frames, a letter from A to Z each, naming its
                          input (default: ${frames_default}).
  -D, --deep RANGES       HIPTRACE_DEEP: hash every buffer argument after the
                          launches of these comma-separated ranges A-B (0-based
                          ordinals up to 18446744073709551615, B excluded) and
                          ordinals A, such as 0-10,20, or after every launch
                          with all (default: unset; none).
  -k, --deep-kernels LIST HIPTRACE_DEEP_KERNELS: hash the buffer arguments of
                          every launch of these comma-separated kernels;
                          c32_post_merge_head_half gives the network's output
                          (default: unset; none).
  -h, --help              Show this help and exit (default: off).

Relative paths are resolved against the working directory. HIPTRACE_REAL, when
set, names the real HIP runtime; otherwise the tracing runtime loads the first
that dlsslopd itself would.
HELP
}

while (( $# )); do
    case $1 in
        -h|--help) usage; exit 0 ;;
        -S|--self-test) self_test=1; shift ;;
        -p|--performance) performance=1; shift ;;
        -M|--motion) motion=1; shift ;;
        --*=*)
            [[ ${option_variable[${1%%=*}]} ]] || fail 2 "trace.sh: unknown option: $1 (see --help)"
            set -- "${1%%=*}" "${1#*=}" "${@:2}" ;;
        -[bdmaocntPFDk]?*) set -- "${1:0:2}" "${1:2}" "${@:2}" ;;
        -*)
            [[ ${option_variable[$1]} ]] || fail 2 "trace.sh: unknown option: $1 (see --help)"
            (( $# > 1 )) && [[ $2 ]] || fail 2 "trace.sh: $1 requires a nonempty value"
            printf -v "${option_variable[$1]}" '%s' "$2"
            shift 2 ;;
        *) fail 2 "trace.sh: unexpected argument: $1 (see --help)" ;;
    esac
done
declare -A size=([720]=1280x720 [900]=1600x900 [1080]=1920x1080)
[[ ${size[$tier]} ]] || fail 2 "trace.sh: --tier must be 720, 900 or 1080"
[[ $passes =~ ^[1-9][0-9]*$ ]] || fail 2 "trace.sh: --passes must be a positive number"
if [[ $deep && $deep != all ]]; then
    [[ $deep =~ ^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$ ]] ||
        fail 2 "trace.sh: --deep must be all or comma-separated A-B ranges and ordinals, such as 0-10,20"
    # The tracing runtime refuses an ordinal above 2^64 - 1, and hashes nothing by ordinal then.
    IFS=,- read -r -a ordinals <<< "$deep"
    for ordinal in "${ordinals[@]}"; do
        [[ $ordinal =~ ^0*([0-9]+)$ ]]
        digits=${BASH_REMATCH[1]}
        if (( ${#digits} > 20 )) || { (( ${#digits} == 20 )) && [[ $digits > 18446744073709551615 ]]; }; then
            fail 2 "trace.sh: --deep ordinals must not be above 18446744073709551615"
        fi
    done
fi
if [[ $self_test ]] && [[ $motion || $frames ]]; then
    fail 2 "trace.sh: --motion and --frames apply to served frames, not to --self-test"
fi
if [[ ! $name ]]; then
    name=serve-$tier
    [[ $self_test ]] && name=selftest-$tier
    [[ $performance ]] && name+=-perf
    [[ $motion ]] && name+=-motion
    (( passes > 1 )) && name+=-passes$passes
fi
[[ $name == */* ]] && fail 2 "trace.sh: --name must not contain /"
for variable in build daemon modules assets output channel; do
    [[ ${!variable} && ${!variable} != /* ]] && printf -v "$variable" '%s' "$PWD/${!variable}"
done
daemon=${daemon:-$build/dlsslopd}
[[ -x $daemon ]] || fail 1 "trace.sh: no dlsslopd at $daemon"
[[ -f $build/libhiptrace.so ]] || fail 1 "trace.sh: no libhiptrace.so in $build"
[[ $self_test || -x $build/shmclient ]] || fail 1 "trace.sh: no shmclient in $build"
[[ -d $modules ]] || fail 1 "trace.sh: no module directory $modules"
[[ -d $assets ]] || fail 1 "trace.sh: no model directory $assets"
mkdir -p -- "$output" || fail 1 "trace.sh: cannot create $output"
[[ -d $channel ]] || mkdir -p -m 700 -- "$channel" || fail 1 "trace.sh: cannot create $channel"

command=("$daemon" --config /dev/null --backend hip --tier "$tier" --passes "$passes" --modules "$modules"
         --assets "$assets" --shm "$channel/$name.bin")
[[ $performance ]] && command+=(--performance)
if [[ $self_test ]]; then
    command+=(--self-test)
else
    wh=${size[$tier]}
    client=("$build/shmclient" --shm "$channel/$name.bin" --width "${wh%x*}" --height "${wh#*x}"
            --tier "$tier" --passes "$passes" --frames "${frames:-$frames_default}" --log "$output/$name.daemon.log")
    [[ $motion ]] && client+=(--mvec 1)
    command=("${client[@]}" -- "${command[@]}")
fi
export DLSSLOP_HIP_LIBRARY=$build/libhiptrace.so HIPTRACE_FILE=$output/$name.trace HIPTRACE_MODULES=$modules \
       HIPTRACE_DEEP=$deep HIPTRACE_DEEP_KERNELS=$deep_kernels
"${command[@]}" 2>&1 | tee -- "$output/$name.log"
status=${PIPESTATUS[0]}
printf '%s: status %d; trace %s\n' "$name" "$status" "$HIPTRACE_FILE"
exit "$status"
