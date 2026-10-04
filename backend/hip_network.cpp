// SPDX-License-Identifier: MIT
#include "hip_network.hpp"
#include "files.hpp"

#include <bit>
#include <cstdio>
#include <iterator>

namespace dlsslop::hip {

// A kernel takes each argument's bytes from the start of its 8-byte value.
static_assert(std::endian::native == std::endian::little && sizeof(void*) == sizeof(uint64_t));

Model::~Model()
{
    for (void* weight : weights_) api_.hipFree(weight);
    for (Handle module : modules_)
        if (module) api_.hipModuleUnload(module);
}

Result<void> Model::upload(const std::string& assets, const hip_weight_spec& spec, hip_weight_file& file)
{
    struct error e;
    if (const enum error_code code = hip_weights_load(&file, assets.c_str(), assets.size(), &spec, &e))
        return forward_c(code, e);
    const size_t bytes = file.count * sizeof(float);
    void* weight = nullptr;
    if (const int error = api_.hipMalloc(&weight, bytes))
        return api_.check(error, ("allocate weight " + std::string(file.path)).c_str());
    weights_.push_back(weight);
    bytes_ += bytes;
    if (const int error = api_.hipMemcpy(weight, file.values, bytes, 1))
        return api_.check(error, ("upload weight " + std::string(file.path)).c_str());
    return {};
}

Result<void> Model::load(const std::string& modules, const std::string& assets, const hip_weight_spec* weights,
                         size_t count)
{
    for (size_t m = 0; m < HIP_MODULE_COUNT; ++m)
        modules_[m] = DLSSLOP_TRY(load_module(api_, join(modules, HIP_PLAN_MODULE_FILES[m])));
    for (size_t k = 0; k < HIP_KERNEL_COUNT; ++k)
        DLSSLOP_TRY(api_.check(api_.hipModuleGetFunction(&functions_[k], modules_[HIP_PLAN_KERNELS[k].module],
                                                         HIP_PLAN_KERNELS[k].name),
                               HIP_PLAN_KERNELS[k].name));
    weights_.reserve(count);
    for (size_t w = 0; w < count; ++w) {
        hip_weight_file file{};
        const Result<void> uploaded = upload(assets, weights[w], file);
        hip_weight_file_fini(&file);
        if (!uploaded) return uploaded;
    }
    return {};
}

Network::~Network()
{
    for (void* buffer : buffers_) api_.hipFree(buffer);
    for (void* map : gather_)
        if (map) api_.hipFree(map);
}

Result<void> Network::build(const hip_plan& plan, const hip_placement& placement)
{
    buffers_.reserve(placement.buffer_count);
    for (size_t b = 0; b < placement.buffer_count; ++b) {
        const size_t bytes = placement.buffers[b];
        void* buffer = nullptr;
        DLSSLOP_TRY(api_.check(api_.hipMalloc(&buffer, bytes), "allocate network buffer"));
        buffers_.push_back(buffer);
        bytes_ += bytes;
    }
    std::vector<uint32_t> map(size_t(plan.tokens) * 1024);
    for (unsigned inverse = 0; inverse < 2; ++inverse) {
        hip_plan_gather_map(map.data(), plan.tokens, inverse);
        const size_t bytes = map.size() * sizeof(uint32_t);
        void* buffer = nullptr;
        DLSSLOP_TRY(api_.check(api_.hipMalloc(&buffer, bytes), "allocate gather map"));
        gather_[inverse] = buffer;
        bytes_ += bytes;
        DLSSLOP_TRY(api_.check(api_.hipMemcpy(buffer, map.data(), bytes, 1), "upload gather map"));
    }
    size_t arguments = 0;
    for (size_t l = 0; l < plan.launch_count; ++l) arguments += plan.launches[l].count;
    values_.assign(2 * arguments, 0);
    argv_.assign(2 * arguments, nullptr);
    // The frame's arguments, in the order of their kinds.
    static_assert(HIP_ARG_HISTORY == HIP_ARG_RGBA + 1 && HIP_ARG_TEMPORAL == HIP_ARG_RGBA + 2 &&
                  HIP_ARG_OUTPUT == HIP_ARG_RGBA + 3);
    void* const frame[] = {&frame_.rgba, &frame_.history, &frame_.temporal, &frame_.output};
    launches_.reserve(plan.launch_count);
    size_t at = 0;
    for (size_t l = 0; l < plan.launch_count; ++l) {
        const hip_launch& launch = plan.launches[l];
        launches_.push_back({model_.function(launch.kernel), launch.grid, HIP_PLAN_KERNELS[launch.kernel].threads,
                             launch.kernel, {&argv_[at], &argv_[arguments + at]}});
        for (unsigned later = 0; later < 2; ++later) {
            const uint16_t* const buffer_of = later ? placement.later : placement.first;
            for (unsigned i = 0; i < launch.count; ++i) {
                const hip_arg arg = launch.args[i];
                const size_t slot = later * arguments + at + i;
                uint64_t& value = values_[slot];
                argv_[slot] = &value;
                switch (arg.kind) {
                case HIP_ARG_U32:
                case HIP_ARG_F32: value = arg.value; break;
                case HIP_ARG_NULL: break;
                case HIP_ARG_TENSOR: value = std::bit_cast<uint64_t>(buffers_[buffer_of[arg.value]]); break;
                case HIP_ARG_WEIGHT: value = std::bit_cast<uint64_t>(model_.weight(arg.value)); break;
                case HIP_ARG_GATHER: value = std::bit_cast<uint64_t>(gather_[arg.value]); break;
                case HIP_ARG_RGBA:
                case HIP_ARG_HISTORY:
                case HIP_ARG_TEMPORAL:
                case HIP_ARG_OUTPUT: argv_[slot] = frame[arg.kind - HIP_ARG_RGBA];
                }
            }
        }
        at += launch.count;
    }
    return {};
}

Result<void> Network::enqueue(void* rgba, void* history, void* output)
{
    frame_ = {rgba, history ? history : rgba, output, history != nullptr};
    for (const Bound& b : launches_)
        if (const int error = api_.hipModuleLaunchKernel(b.function, b.grid, 1, 1, b.threads, 1, 1, 0, stream_,
                                                         b.argv[warm_], nullptr))
            return api_.check(error, HIP_PLAN_KERNELS[b.kernel].name);
    warm_ = true;
    return {};
}

Result<void> Network::print_memory() const
{
    size_t free = 0, total = 0;
    DLSSLOP_TRY(api_.check(api_.hipMemGetInfo(&free, &total), "memory stats"));
    std::printf("memory owned_MiB=%.1f allocations=%zu device_free_MiB=%.1f total_MiB=%.1f\n",
                (model_.bytes() + bytes_) / 1048576., model_.count() + buffers_.size() + std::size(gather_),
                free / 1048576., total / 1048576.);
    return {};
}

} // namespace dlsslop::hip
