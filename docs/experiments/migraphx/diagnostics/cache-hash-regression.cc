// Standalone diagnostic; does not modify or load a GPU runtime.
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

#include "core/framework/murmurhash3.h"
#include "migraphx_cache_key.h"

template <typename T>
std::uint64_t shape_hash(const T& value, bool corrected) {
  std::array<std::uint32_t, 4> hash{};
  const auto bytes = value.size() * (corrected ? sizeof(*value.data()) : 1);
  // Match the provider's seed and truncation to the first 64 bits.
  // Supply non-null storage for the empty-input diagnostic.
  const char empty = 0;
  const void* data = value.empty() ? static_cast<const void*>(&empty) : value.data();
  onnxruntime::MurmurHash3::x86_128(data, bytes, hash[0], hash.data());
  return hash[0] | (static_cast<std::uint64_t>(hash[1]) << 32);
}

int main() {
  const std::vector<std::int64_t> t128{1, 128, 128, 1};
  const std::vector<std::int64_t> t256{1, 128, 256, 1};
  const std::vector<std::int64_t> batch2{2, 128, 128, 1};
  const std::vector<std::int64_t> empty;
  assert(shape_hash(t128, false) == shape_hash(t256, false));
  assert(shape_hash(t128, true) != shape_hash(t256, true));
  assert(shape_hash(t128, true) != shape_hash(batch2, true));
  assert(shape_hash(empty, false) == 0);
  assert(shape_hash(std::string_view{"gfx1151"}, false) ==
         shape_hash(std::string_view{"gfx1151"}, true));
  using Inputs = std::map<std::string, std::vector<std::int64_t>>;
  const Inputs inputs128{{"audio_signal", {1, 128, 128}}, {"length", {1}}};
  const Inputs inputs256{{"audio_signal", {1, 128, 256}}, {"length", {1}}};
  const Inputs reverse128{{"length", {1}}, {"audio_signal", {1, 128, 128}}};
  const auto key128 = onnxruntime::MakeMIGraphXShapeCacheKey(inputs128);
  const auto key256 = onnxruntime::MakeMIGraphXShapeCacheKey(inputs256);
  assert(key128 != key256);
  assert(key128 == onnxruntime::MakeMIGraphXShapeCacheKey(reverse128));
  assert(key128.starts_with("shape-v2-"));
  assert(onnxruntime::MakeMIGraphXShapeCacheKey(Inputs{{"x", {1, 2}}, {"y", {3}}}) !=
         onnxruntime::MakeMIGraphXShapeCacheKey(Inputs{{"x", {1}}, {"y", {2, 3}}}));
  assert(onnxruntime::MakeMIGraphXShapeCacheKey(Inputs{{"x", {}}}) !=
         onnxruntime::MakeMIGraphXShapeCacheKey(Inputs{{"x", {1}}}));
  assert(onnxruntime::MakeMIGraphXShapeCacheKey(Inputs{{"x", {1}}}) !=
         onnxruntime::MakeMIGraphXShapeCacheKey(Inputs{{"renamed", {1}}}));
  std::cout << std::hex
            << "current T128 hash: " << shape_hash(t128, false) << '\n'
            << "current T256 hash: " << shape_hash(t256, false) << '\n'
            << "corrected T128 hash: " << shape_hash(t128, true) << '\n'
            << "corrected T256 hash: " << shape_hash(t256, true) << '\n'
            << "empty first-run hash: " << shape_hash(empty, false) << '\n'
            << "versioned T128 key: " << key128 << '\n'
            << "versioned T256 key: " << key256 << '\n'
            << "CACHE HASH REGRESSION: PASS (collision reproduced; byte-count fix distinguishes tested shapes)\n";
}
