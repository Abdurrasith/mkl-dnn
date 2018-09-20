/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include "mkldnn_thread.hpp"

#if MKLDNN_THR == MKLDNN_THR_EIGEN
#include <stdlib.h>
#include <pthread.h>

#include <mutex>

namespace mkldnn {
namespace impl {

Eigen::ThreadPoolInterface &eigenTp() {
    static Eigen::ThreadPoolInterface *eigenTp_;

    if (eigenTp_ == nullptr) {
        static std::mutex mtx;
        std::lock_guard<std::mutex> lock(mtx);

        volatile static bool initialized = false;
        if (!initialized) {
            const char *ont = getenv("OMP_NUM_THREADS");
            int nthr = (int)std::thread::hardware_concurrency();
            if (ont) nthr = atoi(ont);
            if (nthr < 1) nthr = 1;

            // a workaround for IntelCaffe that sets sched mask
            // for the master thread, which makes the whole Eigen
            // thread pool belong to core #0
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            for (int i = 0; i < nthr; ++i)
                CPU_SET(i, &cpuset);
            sched_setaffinity(0, sizeof(cpuset), &cpuset);

            eigenTp_ = new Eigen::ThreadPool(nthr);
            initialized = true;
        }
    }

    return *eigenTp_;
}

}
}

#endif
