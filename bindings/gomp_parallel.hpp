// gomp_parallel_for: the parallel loop the batch runner and the C++ search share.
#pragma once

#include <algorithm>
#include <cstdint>

// The batch runner's parallel loops go through libgomp's own entry point, not `#pragma omp`.
//
// Every process that steps the engine also runs torch, and torch ships libgomp. The dynamic
// loader shares a library by soname, so an extension that needs `libgomp.so.1` gets the very
// copy torch loaded: one OpenMP runtime, one thread pool. `#pragma omp` ties that to the
// compiler -- GCC emits GOMP_* calls, clang emits __kmpc_* calls into LLVM's libomp, and clang
// with `-fopenmp=libgomp` silently emits *serial* code -- so under clang the engine brought a
// second pool of spinning workers next to torch's and lost 5-7% on a rollout loop even though
// its own code was ~20% faster. Calling GOMP_parallel directly keeps the single pool under any
// compiler. It is the ABI every GCC-compiled OpenMP binary calls, so it cannot change under us.
// tests/bindings/test_build_toolchain.py fails if a second pool ever comes back.
extern "C" {
void GOMP_parallel(void (*fn)(void*), void* data, unsigned num_threads, unsigned flags);
int omp_get_thread_num(void);
int omp_get_num_threads(void);
}

namespace ts_parallel {

// `for (i = 0; i < n; ++i) body(i)` across the OpenMP team, split into contiguous equal chunks
// -- what `schedule(static)` does. Each iteration must be independent (they are: one env each).
template <class F>
void gomp_parallel_for(int64_t n, F&& body) {
    struct Job {
        int64_t n;
        F* body;
    };
    Job job{n, &body};
    GOMP_parallel(
        [](void* p) {
            const Job* j = static_cast<const Job*>(p);
            const int64_t team = omp_get_num_threads();
            const int64_t chunk = (j->n + team - 1) / team;
            const int64_t lo = static_cast<int64_t>(omp_get_thread_num()) * chunk;
            const int64_t hi = std::min(j->n, lo + chunk);
            for (int64_t i = lo; i < hi; ++i) (*j->body)(i);
        },
        &job, /*num_threads=*/0 /* OMP_NUM_THREADS / the team torch sized */, /*flags=*/0);
}

}  // namespace ts_parallel

using ts_parallel::gomp_parallel_for;
