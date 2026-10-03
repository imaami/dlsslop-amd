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

Result<void> Model::load(const std::string& modules, const std::string& assets, std::span<const WeightSpec> weights)
{
    for (size_t m = 0; m < size_t(Module::kCount); ++m)
        modules_[m] = DLSSLOP_TRY(load_module(api_, join(modules, kModuleFiles[m])));
    for (size_t k = 0; k < size_t(Kernel::kCount); ++k)
        DLSSLOP_TRY(api_.check(api_.hipModuleGetFunction(&functions_[k], modules_[size_t(kKernels[k].module)],
                                                         kKernels[k].name),
                               kKernels[k].name));
    weights_.reserve(weights.size());
    for (const WeightSpec& spec : weights) {
        auto file = DLSSLOP_TRY(read_weights(assets, spec));
        DLSSLOP_TRY(pack(spec, file));
        const size_t bytes = file.values.size() * sizeof(float);
        void* weight = nullptr;
        if (const int error = api_.hipMalloc(&weight, bytes))
            return api_.check(error, ("allocate weight " + file.path).c_str());
        weights_.push_back(weight);
        bytes_ += bytes;
        if (const int error = api_.hipMemcpy(weight, file.values.data(), bytes, 1))
            return api_.check(error, ("upload weight " + file.path).c_str());
    }
    return {};
}

Network::~Network()
{
    for (void* buffer : buffers_) api_.hipFree(buffer);
    for (void* map : gather_)
        if (map) api_.hipFree(map);
}

Result<void> Network::build(const Plan& plan, const Placement& placement)
{
    buffers_.reserve(placement.buffers.size());
    for (const size_t bytes : placement.buffers) {
        void* buffer = nullptr;
        DLSSLOP_TRY(api_.check(api_.hipMalloc(&buffer, bytes), "allocate network buffer"));
        buffers_.push_back(buffer);
        bytes_ += bytes;
    }
    for (unsigned inverse = 0; inverse < 2; ++inverse) {
        const std::vector<uint32_t> map = gather_map(plan.tokens, inverse);
        const size_t bytes = map.size() * sizeof(uint32_t);
        void* buffer = nullptr;
        DLSSLOP_TRY(api_.check(api_.hipMalloc(&buffer, bytes), "allocate gather map"));
        gather_[inverse] = buffer;
        bytes_ += bytes;
        DLSSLOP_TRY(api_.check(api_.hipMemcpy(buffer, map.data(), bytes, 1), "upload gather map"));
    }
    size_t arguments = 0;
    for (const Launch& launch : plan.launches) arguments += launch.count;
    values_.assign(2 * arguments, 0);
    argv_.assign(2 * arguments, nullptr);
    // The frame's arguments, in the order of their kinds.
    static_assert(Arg::kHistory == Arg::kRgba + 1 && Arg::kTemporal == Arg::kRgba + 2 &&
                  Arg::kOutput == Arg::kRgba + 3);
    void* const frame[] = {&frame_.rgba, &frame_.history, &frame_.temporal, &frame_.output};
    launches_.reserve(plan.launches.size());
    size_t at = 0;
    for (const Launch& launch : plan.launches) {
        launches_.push_back({model_.function(launch.kernel), launch.grid, kKernels[size_t(launch.kernel)].threads,
                             launch.kernel, {&argv_[at], &argv_[arguments + at]}});
        for (unsigned later = 0; later < 2; ++later) {
            const std::vector<uint16_t>& buffer_of = later ? placement.later : placement.first;
            for (unsigned i = 0; i < launch.count; ++i) {
                const Arg arg = launch.args[i];
                const size_t slot = later * arguments + at + i;
                uint64_t& value = values_[slot];
                argv_[slot] = &value;
                switch (arg.kind) {
                case Arg::kU32:
                case Arg::kF32: value = arg.value; break;
                case Arg::kNull: break;
                case Arg::kTensor: value = std::bit_cast<uint64_t>(buffers_[buffer_of[arg.value]]); break;
                case Arg::kWeight: value = std::bit_cast<uint64_t>(model_.weight(arg.value)); break;
                case Arg::kGather: value = std::bit_cast<uint64_t>(gather_[arg.value]); break;
                case Arg::kRgba:
                case Arg::kHistory:
                case Arg::kTemporal:
                case Arg::kOutput: argv_[slot] = frame[arg.kind - Arg::kRgba];
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
            return api_.check(error, kKernels[size_t(b.kernel)].name);
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
