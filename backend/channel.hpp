// The daemon's side of the shared-memory channel.
// SPDX-License-Identifier: MIT
#pragma once
#include "files.hpp"
#include "result.hpp"
#include "shm_protocol.h"

#include <string>

namespace dlsslop {
// The header and the two frame slots, mapped from a private file that one
// daemon locks at a time.
class Mapping {
    Descriptor file_;
    Mapping(Descriptor file, void* mapping);
public:
    ShmHeader* h = nullptr;
    uint8_t* input = nullptr;
    uint8_t* output = nullptr;

    static Result<Mapping> open(const std::string& name);
    Mapping(Mapping&& other) noexcept;
    ~Mapping();
    void reason(const std::string& text)
    {
        ShmStoreString(&h->helperReasonSeq, h->helperReason, kReasonBytes, text.c_str());
    }
};
} // namespace dlsslop
