/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef MKLDNN_THREAD_HPP
#define MKLDNN_THREAD_HPP

#include "utils.hpp"
#include "z_magic.hpp"

#define MKLDNN_THR_SEQ 0
#define MKLDNN_THR_OMP 1
#define MKLDNN_THR_TBB 2
#define MKLDNN_THR_EIGEN 3
#define MKLDNN_THR_TENSORFLOW 4

/* Ideally this condition below should never happen (if the library is built
 * using regular cmake). For the 3rd-party projects that build the library
 * from the sources on their own try to guess the right threading... */
#if !defined(MKLDNN_THR)
#   if defined(_OPENMP)
#       define MKLDNN_THR MKLDNN_THR_OMP
#   else
#       define MKLDNN_THR MKLDNN_THR_SEQ
#   endif
#endif

#if MKLDNN_THR == MKLDNN_THR_SEQ
#define MKLDNN_THR_SYNC 1
inline int mkldnn_get_max_threads() { return 1; }
inline int mkldnn_get_num_threads() { return 1; }
inline int mkldnn_get_thread_num() { return 0; }
inline int mkldnn_in_parallel() { return 0; }
inline void mkldnn_thr_barrier() {}

#elif MKLDNN_THR == MKLDNN_THR_OMP
#include <omp.h>
#define MKLDNN_THR_SYNC 1

inline int mkldnn_get_max_threads() { return omp_get_max_threads(); }
inline int mkldnn_get_num_threads() { return omp_get_num_threads(); }
inline int mkldnn_get_thread_num() { return omp_get_thread_num(); }
inline int mkldnn_in_parallel() { return omp_in_parallel(); }
inline void mkldnn_thr_barrier() {
#   pragma omp barrier
}

#elif MKLDNN_THR == MKLDNN_THR_TBB
#include "tbb/task_arena.h"
#include "tbb/parallel_for.h"

#include "mkldnn.h"
#define MKLDNN_THR_SYNC 0
namespace thr_ns = tbb;

namespace mkldnn {
namespace impl {
// temporary workaround
void MKLDNN_API tbb_init();
}
}

inline int mkldnn_get_max_threads()
{ mkldnn::impl::tbb_init(); return tbb::this_task_arena::max_concurrency(); }
inline int mkldnn_get_num_threads() { return mkldnn_get_max_threads(); }
inline int mkldnn_get_thread_num()
{ mkldnn::impl::tbb_init(); return tbb::this_task_arena::current_thread_index(); }
inline int mkldnn_in_parallel() { mkldnn::impl::tbb_init(); return 0; }
inline void mkldnn_thr_barrier() { assert(!"no barrier in TBB"); }

#elif MKLDNN_THR == MKLDNN_THR_EIGEN \
                   || MKLDNN_THR == MKLDNN_THR_TENSORFLOW

#include "mkldnn.h"

#include "unsupported/Eigen/CXX11/ThreadPool"

#if MKLDNN_THR == MKLDNN_THR_EIGEN
using extern_thread_pool_t = Eigen::ThreadPoolInterface;
#elif MKLDNN_THR == MKLDNN_THR_TENSORFLOW
#pragma push_macro("CHECK")
#undef CHECK
#include "tensorflow/core/lib/core/threadpool.h"
#pragma pop_macro("CHECK")
using extern_thread_pool_t = tensorflow::thread::ThreadPool;
#else
#error unknown MKLDNN_THR
#endif

#define MKLDNN_THR_SYNC 0
namespace thr_ns = Eigen;

namespace mkldnn {
namespace impl {
// temporary workaround
extern_thread_pool_t MKLDNN_API &eigenTp();
}
}

inline int mkldnn_get_max_threads()
{ return mkldnn::impl::eigenTp().NumThreads(); }
inline int mkldnn_get_num_threads() { return mkldnn_get_max_threads(); }
inline int mkldnn_get_thread_num()
{ return mkldnn::impl::eigenTp().CurrentThreadId(); }
inline int mkldnn_in_parallel() { return mkldnn_get_thread_num() != -1; }
inline void mkldnn_thr_barrier() { assert(!"no barrier in Eigen"); }

namespace Eigen {
template <typename F>
void parallel_for(int start, int end, F f) {
    if (end - start == 1) {
        f(start);
        return;
    }

    Eigen::Barrier b(end - start);
    for (int i = start; i < end; ++i) {
        mkldnn::impl::eigenTp().Schedule([i, &f, &b]() { f(i); b.Notify(); });
    }
    b.Wait();
}
} // namespace Eigen
#endif

/* MSVC still supports omp 2.0 only */
#if defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#   define collapse(x)
#   define PRAGMA_OMP_SIMD(...)
#else
#   define PRAGMA_OMP_SIMD(...) PRAGMA_MACRO(CHAIN2(omp, simd __VA_ARGS__))
#endif // defined(_MSC_VER) && !defined(__INTEL_COMPILER)

namespace mkldnn {
namespace impl {

inline bool mkldnn_thr_syncable() { return MKLDNN_THR_SYNC == 1; }

template <typename T, typename U>
inline void balance211(T n, U team, U tid, T &n_start, T &n_end) {
    T n_min = 1;
    T &n_my = n_end;
    if (team <= 1 || n == 0) {
        n_start = 0;
        n_my = n;
    } else if (n_min == 1) {
        // team = T1 + T2
        // n = T1*n1 + T2*n2  (n1 - n2 = 1)
        T n1 = utils::div_up(n, (T)team);
        T n2 = n1 - 1;
        T T1 = n - n2 * (T)team;
        n_my = (T)tid < T1 ? n1 : n2;
        n_start = (T)tid <= T1 ? tid * n1 : T1 * n1 + ((T)tid - T1) * n2;
    }

    n_end += n_start;
}

} // namespace impl
} // namespace mkldnn

#include "mkldnn_thread_parallel_nd.hpp"

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
