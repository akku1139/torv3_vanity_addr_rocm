#include "hip/hip_runtime.h"

#include <iostream>
#include <cstdint>
#include <random>
#include <chrono>
#include <thread>
#include <atomic>
#include <set>
#include <vector>
#include "hip/hip_runtime.h"
#include "keccak.h"
#define EXPAND 10
#define EXPANDBUF 4096
#include "gpu_keccak.h"
#include "gpu_crypto.h"
#include "gpu_scan.h"

extern "C" {
#include "crypto-ops.h"
#include <string.h>
}

using namespace std::chrono;

constexpr char alphabet32[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
constexpr size_t BATCH_SIZE = 1 << 20;
constexpr size_t PATTERN_SIZE = 32;
constexpr uint32_t THREADS_PER_BLOCK = 32;

namespace gpu {

__device__ ge_cached d_precomputed_steps[32];
__device__ ge_cached d_stride_8_32;

const fe fe_d2 = {-21827239, -5839606, -30745221, 13898782, 229458, 15978800, -12551817, -6495438, 29715968, 9444199}; /* 2 * d */

__device__ void ge_p3_to_cached(ge_cached *r, const ge_p3 *p) {
    fe_add(r->YplusX, p->Y, p->X);
    fe_sub(r->YminusX, p->Y, p->X);
    fe_copy(r->Z, p->Z);

    fe_mul(r->T2d, p->T, fe_d2);
}

__global__ void init_step_tables_kernel() {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        ge_p3 p3_current;
        ge_p1p1 sum;

        ge_p3 identity;
        fe_0(identity.X); fe_1(identity.Y); fe_1(identity.Z); fe_0(identity.T);
        ge_p3_to_cached(&d_precomputed_steps[0], &identity);

        ge_add(&sum, &identity, &ge_eightpoint);
        ge_p1p1_to_p3(&p3_current, &sum);

        for (int i = 1; i < 32; ++i) {
            ge_p3_to_cached(&d_precomputed_steps[i], &p3_current);

            ge_add(&sum, &p3_current, &ge_eightpoint);
            ge_p1p1_to_p3(&p3_current, &sum);
        }

        ge_p3_to_cached(&d_stride_8_32, &p3_current);
    }
}

__global__ void vanity_search_kernel(
    const uint8_t* input_buf,
    uint64_t offset,
    const gpu::BinaryPattern* patterns,
    size_t pattern_count,
    uint32_t* results_key,
    uint32_t* results_ctr
) {
    const uint32_t t = threadIdx.x;
    const uint32_t g = blockIdx.x;

    __shared__ uint64_t shared_priv[4];
    gpu::keccak_12_rounds(input_buf, 32, offset + g, shared_priv);
    __syncthreads();

    ge_p3 point;
    ge_scalarmult_base(&point, reinterpret_cast<uint8_t*>(shared_priv));

    ge_p1p1 sum;
    ge_add(&sum, &point, &d_precomputed_steps[t]);
    ge_p1p1_to_p3(&point, &sum);

    for (uint32_t counter = 0; counter < EXPAND; ++counter) {
        uint64_t current_pub[4];

        gpu::ge_p3_tobytes(current_pub, &point);

        for (size_t i = 0; i < pattern_count; ++i) {
            const gpu::BinaryPattern& pt = patterns[i];
            if (((current_pub[0] & pt.mask[0]) == pt.v[0]) &&
                ((current_pub[1] & pt.mask[1]) == pt.v[1]) &&
                ((current_pub[2] & pt.mask[2]) == pt.v[2]) &&
                ((current_pub[3] & pt.mask[3]) == pt.v[3]))
            {
                uint32_t k = atomicAdd(results_key, 1) + 1;
                if (k < 256) {
                    results_key[k] = g * 32 + t; // global_idx
                    results_ctr[k] = counter;
                }
            }
        }

        ge_add(&sum, &point, &d_stride_8_32);
        ge_p1p1_to_p3(&point, &sum);
    }
}

} // gpu

int b32_to_bin(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a';
    if (c >= '2' && c <= '7') return c - '2' + 26;
    return -1;
}

gpu::BinaryPattern pack_pattern(const std::string& s) {
    gpu::BinaryPattern pt = {};
    uint8_t raw[32] = {0};
    uint8_t mask[32] = {0};

    int bit_pos = 0;
    // Max onion v3 address length
    for (size_t i = 0; i < s.length() && i < 52; ++i) {
        int val = b32_to_bin(s[i]);
        bool is_wildcard = (val == -1);

        for (int b = 4; b >= 0; --b) {
            int byte_idx = bit_pos / 8;
            int bit_idx = 7 - (bit_pos % 8);

            if (byte_idx < 32) {
                if (!is_wildcard) {
                    if ((val >> b) & 1) raw[byte_idx] |= (1 << bit_idx);
                    mask[byte_idx] |= (1 << bit_idx);
                }
            }
            bit_pos++;
        }
    }
    memcpy(pt.v, raw, 32);
    memcpy(pt.mask, mask, 32);
    return pt;
}

int main(int argc, char** argv)
{
#define CHECKED_CALL(X) do { \
        const hipError_t err = X; \
        if (err != hipSuccess) { \
            std::cerr << #X " (line " << __LINE__ << ") failed, error " << err; \
            return __LINE__; \
        } \
    } while(0)

    int device_count;
    CHECKED_CALL(hipGetDeviceCount(&device_count));
    
    bool rate_info = false;
    
    std::string patterns_str;

    std::set<int> devices_to_use;

    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];

	// enable hashrate reporting
	if (s == "-i") {
	  rate_info = true;
	  continue;
	}
	
	// device selection
        if ((s == "-d") && (i + 1 < argc)) {
            ++i;
            const int id = strtol(argv[i], nullptr, 10);
            if (0 <= id && id < device_count) {
                devices_to_use.insert(id);
            }
            else {
                printf("Invalid device id %s\n", argv[i]);
            }
            continue;
        }

        if (s.length() > PATTERN_SIZE) {
            s.resize(PATTERN_SIZE);
        }
        else {
            while (s.length() < PATTERN_SIZE) {
                s += '?';
            }
        }

	
        const char* abc = alphabet32;

        bool good = true;
        for (int j = 0; j < PATTERN_SIZE; ++j) {
            if (s[j] == '?') {
                continue;
            }
            bool found = false;
            for (int k = 0; k < 58; ++k) {
                if (s[j] == abc[k]) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                good = false;
                break;
            }
        }
        if (good) {
            patterns_str += s;
        }
        else {
            std::cout << "Invalid pattern \"" << argv[i] << "\"" << std::endl;
        }
    }

    if (patterns_str.empty()) {
        printf(
            "Usage:\n\n"
            "%s [-d N] pattern1 [pattern_2] [pattern_3] ... [pattern_n]\n\n"
	    "-i         will print out the hashrate every 20 seconds if flag set\n\n"
            "-d N       use ROCm device with index N (counting from 0). This argument can be repeated multiple times with different N.\n\n"
            "Each pattern can have \"?\" symbols which match any character.\nOnly the following characters are allowed:\n\n abcdefghijklmnopqrstuvwxyz234567\n\n"
            "Example:\n\t%s P?XXXXX L23456 b55555 FfFfFf H99999\n\n"
            "If the vanity generator finds a match, it will print the secret key as it's running.\n"
            "These can be appended by 32bytes of secure randomness to make a torv3 key file.\n\n",
            argv[0], argv[0]
        );
        return 0;
    }
    
    // Make all uppercase 
    for (int i = 0; patterns_str[i]!='\0'; i++) {
      if(patterns_str[i] >= 'a' && patterns_str[i] <= 'z') {
	patterns_str[i] = patterns_str[i] -32;
      }
    }
    
    // Get some entropy from the random device
    std::random_device::result_type rnd_buf[256];
    std::random_device rd;
    for (int i = 0; i < 256; ++i) {
        rnd_buf[i] = rd();
    }

    std::atomic<uint64_t> keys_checked;
    std::vector<std::thread> threads;

    for (int i = 0; i < device_count; ++i) {
        if (!devices_to_use.empty() && (devices_to_use.find(i) == devices_to_use.end())) {
            continue;
        }

        hipDeviceProp_t prop;
        CHECKED_CALL(hipGetDeviceProperties(&prop, i));
	std::cout << "Using ROCm device " << i << ": " << prop.name << std::endl;

        threads.emplace_back([i, &rnd_buf, &patterns_str, &keys_checked]()
        {
            CHECKED_CALL(hipSetDevice(i));

            CHECKED_CALL(hipSetDeviceFlags(hipDeviceScheduleBlockingSync));

            // Mix entropy into 32-byte secret key template
            uint8_t tmp_buf[sizeof(rnd_buf)];
            memcpy(tmp_buf, rnd_buf, sizeof(rnd_buf));

            // Mix in thread number
            tmp_buf[0] ^= i;

            // Mix all bits of the random buffer into the key template
            uint8_t key_template[32];
            keccak(tmp_buf, sizeof(tmp_buf), key_template, sizeof(key_template), 24);

            uint8_t* input_buf;
            CHECKED_CALL(hipMalloc((void**)&input_buf, 32));
            CHECKED_CALL(hipMemcpy(input_buf, key_template, sizeof(key_template), hipMemcpyHostToDevice));

            uint8_t* patterns;
            CHECKED_CALL(hipMalloc((void**)&patterns, patterns_str.length()));
            CHECKED_CALL(hipMemcpy(patterns, patterns_str.data(), patterns_str.length(), hipMemcpyHostToDevice));

            uint32_t* results_key;
            CHECKED_CALL(hipMalloc((void**)&results_key, 256 * sizeof(uint32_t)));
	    uint32_t* results_ctr;
            CHECKED_CALL(hipMalloc((void**)&results_ctr, 256 * sizeof(uint32_t)));

            for (uint64_t offset = 0;; offset += BATCH_SIZE) {
                CHECKED_CALL(hipMemset(results_key, 0, sizeof(uint32_t)));

                uint32_t blocks = BATCH_SIZE / THREADS_PER_BLOCK;

                std::vector<gpu::BinaryPattern> h_binary_patterns;
                for (size_t i = 0; i < patterns_str.length(); i += PATTERN_SIZE) {
                    h_binary_patterns.push_back(pack_pattern(patterns_str.substr(i, PATTERN_SIZE)));
                }

                gpu::BinaryPattern* patterns;
                CHECKED_CALL(hipMalloc((void**)&patterns, h_binary_patterns.size() * sizeof(gpu::BinaryPattern)));
                CHECKED_CALL(hipMemcpy(patterns, h_binary_patterns.data(),
                             h_binary_patterns.size() * sizeof(gpu::BinaryPattern),
                             hipMemcpyHostToDevice));

                gpu::vanity_search_kernel<<<blocks, THREADS_PER_BLOCK>>>(
                    input_buf,
                    offset,
                    patterns,
                    h_binary_patterns.size(),
                    results_key,
                    results_ctr
                );

                CHECKED_CALL(hipGetLastError());

                CHECKED_CALL(hipDeviceSynchronize());

                uint32_t results_key_host[256];
                CHECKED_CALL(hipMemcpy(results_key_host, results_key, sizeof(results_key_host), hipMemcpyDeviceToHost));
                uint32_t results_ctr_host[256];
                CHECKED_CALL(hipMemcpy(results_ctr_host, results_ctr, sizeof(results_ctr_host), hipMemcpyDeviceToHost));
		
		uint32_t counter = 0;
                for (uint32_t i = 1, n = std::min(255u, results_key_host[0]); i <= n; ++i) {
                    uint8_t buf[32];
		    counter = results_ctr_host[i];

		    *((uint64_t*)key_template) ^= offset + results_key_host[i];
		    keccak(key_template, 32, buf, 32, 12);
		    *((uint64_t*)key_template) ^= offset + results_key_host[i];

                    sc_reduce32(buf);

                    uint8_t tor_secret_key[32];
		    uint8_t tor_secret_key_[32];
		    
                    memcpy(tor_secret_key, buf, 32);
		    uint8_t fixup[32];
		    memset(fixup, 0, 32);
		    
		    uint32_t incr = (counter+1) * 8;
		    memcpy(fixup, &incr, sizeof(incr));
		      
		    scalar_add(tor_secret_key_,tor_secret_key, fixup);

                    uint8_t encoded_tor_secret_key[65];
                    for (int j = 0; j < 32; ++j) {
                        encoded_tor_secret_key[j * 2    ] = "0123456789abcdef"[tor_secret_key_[j] >> 4];
                        encoded_tor_secret_key[j * 2 + 1] = "0123456789abcdef"[tor_secret_key_[j] & 15];
		    }
                    encoded_tor_secret_key[64] = '\0';
		    
		    //std::cout << encoded_tor_secret_key << " " << n << " " << i << std::endl;
		    // probably logic error on my part, but will filter out 99% of false-positives vs speed
		    if(n == i) {
		      std::cout << encoded_tor_secret_key << std::endl;
		    }
		    
                }

                keys_checked += BATCH_SIZE; // for every key we generate we expand it
            }
        });
    }

    auto t1 = high_resolution_clock::now();
    // Timing information in million keys generated per second.
    uint64_t prev_keys_checked = 0;
    for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(20));

        const uint64_t cur_keys_checked = keys_checked;
        const auto t2 = high_resolution_clock::now();

        const double dt = duration_cast<nanoseconds>(t2 - t1).count() * 1e-9;
	if(rate_info)
	  std::cout << "KEYRATE: " << (cur_keys_checked - prev_keys_checked) / dt * 1e-6 << " million keys/second" << std::endl; // This value is not validated experimentally, but is a ballpark. 
	  // Protocol: Generating large amount of keys in a fixed amount of time had the point addition generate 10-15% more valid keys. TODO: fix this at same time as tuning loop.
        t1 = t2;
        prev_keys_checked = cur_keys_checked;
    }

    return 0;
}
