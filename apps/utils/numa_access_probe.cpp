// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <numa.h>
#include <numeric>
#include <random>
#include <vector>
#include <x86intrin.h>

namespace
{
struct Config
{
    size_t num_nodes = 1 << 20;
    size_t iterations = 8;
    size_t coord_bytes = 128;
    size_t nhood_bytes = 128;
    size_t warmup_iterations = 2;
    size_t prefetch_distance = 0;
    uint32_t seed = 42;
    int memory_node = 2;
};

struct BenchResult
{
    uint64_t cycles = 0;
    uint64_t valid_samples = 0;
    uint64_t migrated_samples = 0;
    uint64_t checksum = 0;
};

void usage(const char *prog)
{
    std::cout << "Usage:\n  " << prog
              << " <num_nodes> <iterations> [coord_bytes=128] [nhood_bytes=128]"
                 " [warmup_iterations=2] [prefetch_distance=0] [seed=42] [memory_node=2]\n\n"
                 "Recommended run (remote memory):\n"
                 "  numactl -N 0 "
              << prog << " 1000000 16 128 128 2 0 42 2\n";
}

bool parse_size(const char *s, size_t &out)
{
    char *end = nullptr;
    errno = 0;
    const unsigned long long val = std::strtoull(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0')
    {
        return false;
    }
    out = static_cast<size_t>(val);
    return true;
}

bool parse_u32(const char *s, uint32_t &out)
{
    size_t tmp = 0;
    if (!parse_size(s, tmp))
    {
        return false;
    }
    if (tmp > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }
    out = static_cast<uint32_t>(tmp);
    return true;
}

bool parse_int(const char *s, int &out)
{
    char *end = nullptr;
    errno = 0;
    const long val = std::strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0')
    {
        return false;
    }
    if (val < std::numeric_limits<int>::min() || val > std::numeric_limits<int>::max())
    {
        return false;
    }
    out = static_cast<int>(val);
    return true;
}

inline uint64_t read_tsc_serialized(unsigned int &cpu_id)
{
    _mm_lfence();
    const uint64_t t = __rdtscp(&cpu_id);
    _mm_lfence();
    return t;
}

inline void prefetch_node(const uint8_t *ptr, size_t node_bytes)
{
    __builtin_prefetch(ptr, 0, 3);
    if (node_bytes > 64)
        __builtin_prefetch(ptr + 64, 0, 3);
    if (node_bytes > 128)
        __builtin_prefetch(ptr + 128, 0, 3);
    if (node_bytes > 192)
        __builtin_prefetch(ptr + 192, 0, 3);
}

uint64_t consume_node_like_cached_beam(const uint8_t *node_ptr, size_t coord_bytes, size_t nhood_bytes)
{
    uint64_t acc = 0;

    const size_t coord_u64 = coord_bytes / sizeof(uint64_t);
    for (size_t i = 0; i < coord_u64; ++i)
    {
        uint64_t v = 0;
        std::memcpy(&v, node_ptr + i * sizeof(uint64_t), sizeof(uint64_t));
        acc += v;
    }
    for (size_t i = coord_u64 * sizeof(uint64_t); i < coord_bytes; ++i)
    {
        acc += node_ptr[i];
    }

    const uint8_t *nhood_ptr = node_ptr + coord_bytes;
    const size_t nhood_u32 = nhood_bytes / sizeof(uint32_t);

    // First pass over neighbors: analogous to compute_dists(node_nbrs, ...)
    for (size_t i = 0; i < nhood_u32; ++i)
    {
        uint32_t v = 0;
        std::memcpy(&v, nhood_ptr + i * sizeof(uint32_t), sizeof(uint32_t));
        acc += v;
    }
    // Second pass over neighbors: analogous to visited/retset loop
    for (size_t i = 0; i < nhood_u32; ++i)
    {
        uint32_t v = 0;
        std::memcpy(&v, nhood_ptr + i * sizeof(uint32_t), sizeof(uint32_t));
        acc ^= static_cast<uint64_t>(v) << 1;
    }
    for (size_t i = nhood_u32 * sizeof(uint32_t); i < nhood_bytes; ++i)
    {
        acc += nhood_ptr[i];
    }

    return acc;
}

BenchResult run_zero_copy(const uint8_t *remote_buf, const std::vector<uint32_t> &visit_order, size_t node_bytes,
                          size_t coord_bytes, size_t nhood_bytes, size_t iterations, size_t prefetch_distance)
{
    volatile uint64_t sink = 0;
    BenchResult out;
    for (size_t it = 0; it < iterations; ++it)
    {
        unsigned int cpu_start = 0, cpu_end = 0;
        const uint64_t t0 = read_tsc_serialized(cpu_start);
        for (size_t i = 0; i < visit_order.size(); ++i)
        {
            if (prefetch_distance > 0 && (i + prefetch_distance) < visit_order.size())
            {
                const uint8_t *pf =
                    remote_buf + static_cast<size_t>(visit_order[i + prefetch_distance]) * node_bytes;
                prefetch_node(pf, node_bytes);
            }

            const uint8_t *node_ptr = remote_buf + static_cast<size_t>(visit_order[i]) * node_bytes;
            sink ^= consume_node_like_cached_beam(node_ptr, coord_bytes, nhood_bytes);
        }
        const uint64_t t1 = read_tsc_serialized(cpu_end);
        if (cpu_start == cpu_end)
        {
            out.cycles += (t1 - t0);
            out.valid_samples++;
        }
        else
        {
            out.migrated_samples++;
        }
    }
    out.checksum = sink;
    return out;
}

BenchResult run_memcpy_then_compute(const uint8_t *remote_buf, const std::vector<uint32_t> &visit_order, size_t node_bytes,
                                    size_t coord_bytes, size_t nhood_bytes, size_t iterations)
{
    std::vector<uint8_t> local_node(node_bytes);
    volatile uint64_t sink = 0;
    BenchResult out;
    for (size_t it = 0; it < iterations; ++it)
    {
        unsigned int cpu_start = 0, cpu_end = 0;
        const uint64_t t0 = read_tsc_serialized(cpu_start);
        for (size_t i = 0; i < visit_order.size(); ++i)
        {
            const uint8_t *node_ptr = remote_buf + static_cast<size_t>(visit_order[i]) * node_bytes;
            std::memcpy(local_node.data(), node_ptr, node_bytes);
            sink ^= consume_node_like_cached_beam(local_node.data(), coord_bytes, nhood_bytes);
        }
        const uint64_t t1 = read_tsc_serialized(cpu_end);
        if (cpu_start == cpu_end)
        {
            out.cycles += (t1 - t0);
            out.valid_samples++;
        }
        else
        {
            out.migrated_samples++;
        }
    }
    out.checksum = sink;
    return out;
}

void fill_data(uint8_t *buf, size_t bytes, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    for (size_t i = 0; i < bytes; ++i)
    {
        buf[i] = static_cast<uint8_t>(dist(rng));
    }
}
} // namespace

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        usage(argv[0]);
        return -1;
    }

    Config cfg;
    if (!parse_size(argv[1], cfg.num_nodes) || !parse_size(argv[2], cfg.iterations))
    {
        usage(argv[0]);
        return -1;
    }
    if (argc > 3 && !parse_size(argv[3], cfg.coord_bytes))
    {
        usage(argv[0]);
        return -1;
    }
    if (argc > 4 && !parse_size(argv[4], cfg.nhood_bytes))
    {
        usage(argv[0]);
        return -1;
    }
    if (argc > 5 && !parse_size(argv[5], cfg.warmup_iterations))
    {
        usage(argv[0]);
        return -1;
    }
    if (argc > 6 && !parse_size(argv[6], cfg.prefetch_distance))
    {
        usage(argv[0]);
        return -1;
    }
    if (argc > 7 && !parse_u32(argv[7], cfg.seed))
    {
        usage(argv[0]);
        return -1;
    }
    if (argc > 8 && !parse_int(argv[8], cfg.memory_node))
    {
        usage(argv[0]);
        return -1;
    }

    if (cfg.num_nodes == 0 || cfg.iterations == 0)
    {
        std::cerr << "num_nodes and iterations must be > 0.\n";
        return -1;
    }

    const size_t node_bytes = cfg.coord_bytes + cfg.nhood_bytes;
    if (node_bytes == 0)
    {
        std::cerr << "coord_bytes + nhood_bytes must be > 0.\n";
        return -1;
    }

    if (cfg.num_nodes > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        std::cerr << "num_nodes must be <= " << std::numeric_limits<uint32_t>::max() << ".\n";
        return -1;
    }

    if (cfg.num_nodes > (std::numeric_limits<size_t>::max() / node_bytes))
    {
        std::cerr << "Requested allocation size overflows size_t.\n";
        return -1;
    }

    const size_t total_bytes = cfg.num_nodes * node_bytes;
    if (numa_available() == -1)
    {
        std::cerr << "NUMA is not available on this system.\n";
        return -1;
    }

    auto *remote_buf = static_cast<uint8_t *>(numa_alloc_onnode(total_bytes, cfg.memory_node));
    if (remote_buf == nullptr)
    {
        std::cerr << "Failed to allocate " << total_bytes << " bytes on NUMA node " << cfg.memory_node << ".\n";
        return -1;
    }

    fill_data(remote_buf, total_bytes, cfg.seed);

    std::vector<uint32_t> visit_order(cfg.num_nodes);
    std::iota(visit_order.begin(), visit_order.end(), 0U);
    std::mt19937 shuffle_rng(cfg.seed + 1U);
    std::shuffle(visit_order.begin(), visit_order.end(), shuffle_rng);

    // Warm-up to avoid measuring first-touch/page-fault effects.
    (void)run_zero_copy(remote_buf, visit_order, node_bytes, cfg.coord_bytes, cfg.nhood_bytes, cfg.warmup_iterations, 0);
    (void)run_memcpy_then_compute(remote_buf, visit_order, node_bytes, cfg.coord_bytes, cfg.nhood_bytes,
                                  cfg.warmup_iterations);

    const auto zc = run_zero_copy(remote_buf, visit_order, node_bytes, cfg.coord_bytes, cfg.nhood_bytes, cfg.iterations,
                                  0);
    const auto zc_pf = run_zero_copy(remote_buf, visit_order, node_bytes, cfg.coord_bytes, cfg.nhood_bytes,
                                     cfg.iterations, cfg.prefetch_distance);
    const auto copy = run_memcpy_then_compute(remote_buf, visit_order, node_bytes, cfg.coord_bytes, cfg.nhood_bytes,
                                              cfg.iterations);

    const double zc_nodes = static_cast<double>(zc.valid_samples) * static_cast<double>(cfg.num_nodes);
    const double zc_pf_nodes = static_cast<double>(zc_pf.valid_samples) * static_cast<double>(cfg.num_nodes);
    const double copy_nodes = static_cast<double>(copy.valid_samples) * static_cast<double>(cfg.num_nodes);

    const double zc_cycles_per_node = zc_nodes > 0.0 ? (static_cast<double>(zc.cycles) / zc_nodes) : 0.0;
    const double zc_pf_cycles_per_node = zc_pf_nodes > 0.0 ? (static_cast<double>(zc_pf.cycles) / zc_pf_nodes) : 0.0;
    const double copy_cycles_per_node = copy_nodes > 0.0 ? (static_cast<double>(copy.cycles) / copy_nodes) : 0.0;

    std::cout << "Config\n";
    std::cout << "  num_nodes=" << cfg.num_nodes << ", iterations=" << cfg.iterations
              << ", coord_bytes=" << cfg.coord_bytes << ", nhood_bytes=" << cfg.nhood_bytes
              << ", warmup_iterations=" << cfg.warmup_iterations << ", prefetch_distance=" << cfg.prefetch_distance
              << ", memory_node=" << cfg.memory_node << "\n";
    std::cout << "  total_bytes=" << total_bytes << "\n\n";

    std::cout << "Results (lower is better)\n";
    std::cout << "  zero_copy_cycles_per_node=" << zc_cycles_per_node << ", valid_samples=" << zc.valid_samples
              << ", migrated_samples=" << zc.migrated_samples << ", checksum=" << zc.checksum << "\n";
    std::cout << "  zero_copy_prefetch_cycles_per_node=" << zc_pf_cycles_per_node
              << ", valid_samples=" << zc_pf.valid_samples << ", migrated_samples=" << zc_pf.migrated_samples
              << ", checksum=" << zc_pf.checksum << "\n";
    std::cout << "  memcpy_then_compute_cycles_per_node=" << copy_cycles_per_node
              << ", valid_samples=" << copy.valid_samples << ", migrated_samples=" << copy.migrated_samples
              << ", checksum=" << copy.checksum << "\n";

    if (copy_cycles_per_node > 0.0)
    {
        std::cout << "  speedup(memcpy_then_compute vs zero_copy)=" << (zc_cycles_per_node / copy_cycles_per_node)
                  << "x\n";
    }
    if (copy_cycles_per_node > 0.0)
    {
        std::cout << "  speedup(memcpy_then_compute vs zero_copy_prefetch)="
                  << (zc_pf_cycles_per_node / copy_cycles_per_node) << "x\n";
    }

    numa_free(remote_buf, total_bytes);
    return 0;
}
