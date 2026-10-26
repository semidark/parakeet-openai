// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#include <array>
#include <charconv>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "core/framework/murmurhash3.h"

namespace onnxruntime {

// Names are length-delimited; ranks preserve tensor boundaries. std::map
// provides the same ordering at session creation and during native Run.
// The version prefix prevents loading artifacts made with the old broken key.
inline std::string MakeMIGraphXShapeCacheKey(
    const std::map<std::string, std::vector<std::int64_t>>& inputs) {
  std::string bytes = "inputs=" + std::to_string(inputs.size()) + ";";
  for (const auto& [name, dims] : inputs) {
    bytes += std::to_string(name.size()) + ":" + name + ":";
    bytes += std::to_string(dims.size()) + ":";
    for (const auto dim : dims) {
      bytes += std::to_string(dim) + ",";
    }
    bytes += ";";
  }

  std::array<std::uint32_t, 4> hash{};
  MurmurHash3::x86_128(bytes.data(), bytes.size(), 0, hash.data());
  const auto value = hash[0] | (static_cast<std::uint64_t>(hash[1]) << 32);
  std::array<char, 16> hex{};
  const auto [end, error] = std::to_chars(hex.data(), hex.data() + hex.size(), value, 16);
  (void)error;
  return "shape-v2-" + std::string(hex.data(), end);
}

}  // namespace onnxruntime
