#pragma once
#include <cstdint>

namespace gpu {

struct alignas(16) BinaryPattern {
    uint64_t v[4];
    uint64_t mask[4];
};

__device__ void prefix_scan(
    const BinaryPattern* d_patterns,
    size_t pattern_count,
    const void* kdata,
    uint32_t* results_key,
    uint32_t* results_ctr
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint8_t* base_p = reinterpret_cast<const uint8_t*>(kdata);

    const uint64_t* pub_ptr = reinterpret_cast<const uint64_t*>(base_p + (index * 32 * EXPAND));

    for (uint32_t counter = 0; counter < EXPAND; ++counter) {
        const uint64_t* current_pub = pub_ptr + (counter * 4);

        for (size_t i = 0; i < pattern_count; ++i) {
            const BinaryPattern& pt = d_patterns[i];

            if (((current_pub[0] & pt.mask[0]) == pt.v[0]) &&
                ((current_pub[1] & pt.mask[1]) == pt.v[1]) &&
                ((current_pub[2] & pt.mask[2]) == pt.v[2]) &&
                ((current_pub[3] & pt.mask[3]) == pt.v[3]))
            {
                uint32_t k = atomicAdd(results_key, 1) + 1;
                if (k < 256) {
                    results_key[k] = index;
                    results_ctr[k] = counter;
                }
                return;
            }
        }
    }
}

} // gpu
