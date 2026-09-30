#!/usr/bin/bash
# Runs dlsslopd's Vulkan network once under the tracing Vulkan layer;
# VALIDATION.md describes the procedure.

usage_line='Usage: trace.sh [OPTIONS] [-- CLIENT-OPTIONS...]'
self=$(readlink -f -- "${BASH_SOURCE[0]}")
root=${self%/tests/vktrace/*}
layer_name=VK_LAYER_LOCAL_vktrace
build_default=$root/build
build=$build_default
daemon=
spirv=
model_default=${XDG_DATA_HOME:-$HOME/.local/share}/dlsslop-amd/dlssnr.bin
model=$model_default
output_default=$PWD
output=$output_default
channel_default=${XDG_RUNTIME_DIR:+$XDG_RUNTIME_DIR/dlsslop-vktrace}
channel_default=${channel_default:-/tmp/dlsslop-vktrace-$UID}
channel=$channel_default
name=
tier_default=720
tier=$tier_default
self_test=
passes_default=1
passes=$passes_default
motion=
fp16=
frames_default=AAB
frames=
hash=
hash_submits=
declare -A option_variable=(
    [-b]=build [--build]=build [-d]=daemon [--daemon]=daemon [-s]=spirv [--spirv]=spirv [-m]=model [--model]=model
    [-o]=output [--output]=output [-c]=channel [--channel]=channel [-n]=name [--name]=name [-t]=tier [--tier]=tier
    [-P]=passes [--passes]=passes [-F]=frames [--frames]=frames [-D]=hash [--hash]=hash
    [-u]=hash_submits [--hash-submits]=hash_submits)

fail() {
    printf '%s\n' "$2" >&2
    exit "$1"
}

usage() {
    cat <<HELP
${usage_line}
Run dlsslopd's Vulkan network once under the tracing Vulkan layer from the
build tree, ${layer_name}, which logs the daemon's device work to
OUTPUT/NAME.trace. dlsslopd reads no settings file and uses its own channel in
--channel. Served frames come from shmclient: its output goes to OUTPUT/NAME.log
and the daemon's to OUTPUT/NAME.daemon.log. A self-test's output goes to
OUTPUT/NAME.log and its image to OUTPUT/NAME.ppm. The client's or the
self-test's output is also shown; the daemon's log is only written to the file.
The exit status is the client's or the self-test's; 2 is a usage error.
Arguments after -- go to shmclient, such as its setting options
(--sharpness 0.5, or --sharpness 1:0.5 from frame 1); see shmclient --help.

Options:
  -b, --build DIR         Build tree with libvktrace.so, its manifest in
                          vktrace/ and shmclient (default: ${build_default}).
  -d, --daemon FILE       The dlsslopd to trace (default: BUILD/dlsslopd).
  -s, --spirv DIRS        VKTRACE_SPIRV: ':'-separated directories whose .spv
                          files name the shader modules in the trace
                          (default: BUILD/vulkan-nr/network).
  -m, --model FILE        The Vulkan network's model
                          (default: ${model_default}:
                          XDG_DATA_HOME/dlsslop-amd/dlssnr.bin when XDG_DATA_HOME
                          is set and nonempty, otherwise
                          ~/.local/share/dlsslop-amd/dlssnr.bin).
  -o, --output DIR        Where the trace and logs go; created if missing
                          (default: the working directory,
                          ${output_default}).
  -c, --channel DIR       Private directory for the channel file NAME.bin;
                          created with mode 0700 if missing
                          (default: ${channel_default}:
                          XDG_RUNTIME_DIR/dlsslop-vktrace when XDG_RUNTIME_DIR
                          is set and nonempty, otherwise /tmp/dlsslop-vktrace-\$UID).
  -n, --name NAME         The trace's and logs' name (default: the configuration:
                          serve or selftest, the tier, then -fp16, -motion and
                          -passesN when given, such as serve-720-fp16).
  -t, --tier HEIGHT       Neural raster: 720, 900 or 1080; frames are served at
                          1280x720, 1600x900 or 1920x1080 (default: ${tier_default}).
  -S, --self-test         Run dlsslopd --self-test instead of serving frames
                          (default: off).
  -P, --passes N          Chained evaluations per frame (default: ${passes_default}).
  -M, --motion            Serve with motion history (default: off).
  -H, --fp16              Serve FP16 proxies (default: off; RGBA8).
  -F, --frames SPEC       Served frames, a letter each: A to Z name the
                          self-test's gradient, shifted by the letter, and f a
                          flat colour (default: ${frames_default}).
  -D, --hash SELECTORS    VKTRACE_HASH: comma-separated content hash selectors
                          i2b, copydst, storage, dispatch=N, all, bufN and imgN,
                          hashed after each submission they select
                          (default: unset; none).
  -u, --hash-submits LIST VKTRACE_HASH_SUBMITS: the submissions to hash, as
                          comma-separated 1-based ranges A-B and ordinals A, or
                          dispatch for those that dispatch anything
                          (default: unset; every submission).
  -h, --help              Show this help and exit (default: off).

Relative paths are resolved against the working directory. The layer goes
first in VK_INSTANCE_LAYERS, before any layer the environment already names,
such as VK_LAYER_KHRONOS_validation, which then also checks the layer's own
hashing commands. dlsslopd keeps its pipeline cache in XDG_CACHE_HOME.
HELP
}

client_arguments=()
while (( $# )); do
    case $1 in
        -h|--help) usage; exit 0 ;;
        -S|--self-test) self_test=1; shift ;;
        -M|--motion) motion=1; shift ;;
        -H|--fp16) fp16=1; shift ;;
        --) client_arguments=("${@:2}"); break ;;
        --*=*)
            [[ ${option_variable[${1%%=*}]} ]] || fail 2 "trace.sh: unknown option: $1 (see --help)"
            set -- "${1%%=*}" "${1#*=}" "${@:2}" ;;
        -[bdsmocntPFDu]?*) set -- "${1:0:2}" "${1:2}" "${@:2}" ;;
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
if [[ $self_test ]] && [[ $motion || $fp16 || $frames || ${#client_arguments[@]} -gt 0 ]]; then
    fail 2 "trace.sh: --motion, --fp16, --frames and client options apply to served frames, not to --self-test"
fi
if [[ ! $name ]]; then
    name=serve-$tier
    [[ $self_test ]] && name=selftest-$tier
    [[ $fp16 ]] && name+=-fp16
    [[ $motion ]] && name+=-motion
    (( passes > 1 )) && name+=-passes$passes
fi
[[ $name == */* ]] && fail 2 "trace.sh: --name must not contain /"
for variable in build daemon model output channel; do
    [[ ${!variable} && ${!variable} != /* ]] && printf -v "$variable" '%s' "$PWD/${!variable}"
done
daemon=${daemon:-$build/dlsslopd}
spirv=${spirv:-$build/vulkan-nr/network}
manifests=$build/vktrace
[[ -x $daemon ]] || fail 1 "trace.sh: no dlsslopd at $daemon"
[[ -f $build/libvktrace.so ]] || fail 1 "trace.sh: no libvktrace.so in $build"
[[ -f $manifests/VkLayer_LOCAL_vktrace.json ]] || fail 1 "trace.sh: no layer manifest in $manifests"
[[ $self_test || -x $build/shmclient ]] || fail 1 "trace.sh: no shmclient in $build"
[[ -f $model ]] || fail 1 "trace.sh: no model at $model"
mkdir -p -- "$output" || fail 1 "trace.sh: cannot create $output"
[[ -d $channel ]] || mkdir -p -m 700 -- "$channel" || fail 1 "trace.sh: cannot create $channel"

command=("$daemon" --config /dev/null --backend vulkan --tier "$tier" --passes "$passes" --vulkan-model "$model"
         --shm "$channel/$name.bin")
if [[ $self_test ]]; then
    command+=(--self-test --output "$output/$name.ppm")
else
    wh=${size[$tier]}
    client=("$build/shmclient" --shm "$channel/$name.bin" --width "${wh%x*}" --height "${wh#*x}" --tier "$tier"
            --passes "$passes" --frames "${frames:-$frames_default}" --log "$output/$name.daemon.log")
    [[ $motion ]] && client+=(--mvec 1)
    [[ $fp16 ]] && client+=(--fp16)
    command=("${client[@]}" "${client_arguments[@]}" -- "${command[@]}")
fi
export VK_ADD_LAYER_PATH=$manifests${VK_ADD_LAYER_PATH:+:$VK_ADD_LAYER_PATH} \
       VK_INSTANCE_LAYERS=$layer_name${VK_INSTANCE_LAYERS:+:$VK_INSTANCE_LAYERS} \
       VKTRACE_FILE=$output/$name.trace VKTRACE_SPIRV=$spirv VKTRACE_HASH=$hash VKTRACE_HASH_SUBMITS=$hash_submits
"${command[@]}" 2>&1 | tee -- "$output/$name.log"
status=${PIPESTATUS[0]}
printf '%s: status %d; trace %s\n' "$name" "$status" "$VKTRACE_FILE"
exit "$status"
