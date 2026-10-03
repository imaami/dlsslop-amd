// The HIP network run by dlsslopd's own host code: the model's code objects and
// packed weights, and a tier's pool buffers with every launch of its plan bound
// once, so that a frame is its kernel launches alone.
// SPDX-License-Identifier: MIT
#pragma once
#include "hip.hpp"
#include "hip_plan.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace dlsslop::hip {

// The network's code objects, every kernel of kKernels, and the weights of a
// plan, packed and uploaded in its order.
class Model {
    const Api& api_;
    Handle modules_[size_t(Module::kCount)]{};
    Handle functions_[size_t(Kernel::kCount)]{};
    std::vector<void*> weights_;
    size_t bytes_ = 0;

public:
    explicit Model(const Api& api) : api_(api) {}
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    // Frees what load() made; the stream must be done with it.
    ~Model();
    // The modules in MODULES with their kernels, then each of WEIGHTS read
    // from ASSETS, packed and uploaded. On failure, what it loaded stays for
    // the destructor.
    Result<void> load(const std::string& modules, const std::string& assets, std::span<const WeightSpec> weights);
    Handle function(Kernel kernel) const { return functions_[size_t(kernel)]; }
    // The image of the plan's weight INDEX.
    void* weight(uint32_t index) const { return weights_[index]; }
    // The device memory the weights take, and their allocations.
    size_t bytes() const { return bytes_; }
    size_t count() const { return weights_.size(); }
};

// A tier's network: its pool buffers and gather maps, and each launch of its
// plan with the arguments bound twice, once for the first frame's buffers and
// once for every later frame's.
//
// Within a frame, a pool buffer holds several tensors in turn, as upstream's
// allocator assigns them. That is correct only because every network launch,
// and every kernel or copy of dlsslopd's that touches the network's buffers,
// is ordered on the one stream. A frame's rgba must stay unchanged until the
// network's last kernel has run, since block 0 and the post kernel both read
// it; its history is read only by block 0, and its output is written only by
// the post kernel. The caller keeps all three until the stream is past the
// network. Model::load() and build() make every allocation, on the thread
// that selected the device; a frame makes none.
class Network {
    struct Bound {
        Handle function;
        uint32_t grid;
        uint16_t threads;
        Kernel kernel;
        void** argv[2]; // For the first frame, and for later ones.
    };
    // A frame's own arguments, which the bound arguments point to. HIP copies
    // each argument as a launch is queued, so the next frame may change them.
    struct Frame {
        void* rgba;
        void* history; // The rgba when the frame has no history.
        void* output;
        uint32_t temporal;
    };

    const Api& api_;
    const Model& model_;
    const Handle stream_;
    std::vector<void*> buffers_;
    void* gather_[2]{}; // The ViT's gather map, and its inverse.
    size_t bytes_ = 0;
    std::vector<Bound> launches_;
    // Each argument's value, 8 bytes whatever its size, and its address in argv.
    std::vector<uint64_t> values_;
    std::vector<void*> argv_;
    Frame frame_{};
    bool warm_ = false;

public:
    Network(const Api& api, const Model& model, Handle stream) : api_(api), model_(model), stream_(stream) {}
    Network(const Network&) = delete;
    Network& operator=(const Network&) = delete;
    // Frees what build() made; the stream must be done with it.
    ~Network();
    // The pool PLACEMENT sizes, the gather maps and PLAN's launches, bound.
    // On failure, what it made stays for the destructor.
    Result<void> build(const Plan& plan, const Placement& placement);
    // One evaluation of the W x H x 4 floats at RGBA, with the frame at
    // HISTORY (null for none), writing W x H x 3 floats to OUTPUT; queued on
    // the stream.
    Result<void> enqueue(void* rgba, void* history, void* output);
    // The network's device memory and the device's, to stdout.
    Result<void> print_memory() const;
};

} // namespace dlsslop::hip
