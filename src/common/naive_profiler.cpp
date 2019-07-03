/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include <cassert>
#include <cfloat>

#include <chrono>
#include <mutex>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>
#include <list>

#include <pthread.h>
#include <unistd.h>

#include <sys/syscall.h>

#include "naive_profiler.hpp"

namespace mkldnn {
namespace impl {
namespace timer {

#ifdef MKLDNN_ENABLE_NAIVE_PROFILER

inline int gettid() { return syscall(SYS_gettid); }

struct ProfilingStats {
    using clock = std::chrono::high_resolution_clock;
    using time_point = clock::time_point;
    using intervals_list = std::list<std::pair<time_point, time_point>>;

    static constexpr int max_kind = 16;

    ProfilingStats(int tid): tid_(tid) {}

    bool empty(int kind) { return num_calls_[kind] <= skip_calls_; }
    int tid() { return tid_; }
    long num_calls(int kind) { return std::max(0L, num_calls_[kind] - skip_calls_); }
    double min_time(int kind) { return empty(kind) ? 0.0 : min_time_[kind]; }
    double max_time(int kind) { return empty(kind) ? 0.0 : max_time_[kind]; }
    double tot_time(int kind) { return empty(kind) ? 0.0 : tot_time_[kind]; }
    double avg_time(int kind) { return empty(kind) ? 0.0 : tot_time_[kind] / num_calls(kind); }

    void start_timer(int kind) {
        if (kind < 0 || kind >= max_kind)
            return;
        assert(gettid() == tid_);
        assert(depth_[kind] >= 0);
        if (depth_[kind] == 0)
            timer_start_[kind] = clock::now();
        depth_[kind]++;
    }

    void stop_timer(int kind) {
        if (kind < 0 || kind >= max_kind)
            return;
        assert(gettid() == tid_);

        if (depth_[kind] == 0)
            return;
        depth_[kind]--;
        if (depth_[kind] > 0)
            return;

        time_point timer_stop = clock::now();

        num_calls_[kind]++;
        if (!empty(kind)) {
            time_point timer_start = timer_start_[kind];
            double time = std::chrono::duration<double, std::micro>(
                    timer_stop - timer_start).count();
            tot_time_[kind] += time;
            min_time_[kind] = std::min(min_time_[kind], time);
            max_time_[kind] = std::max(max_time_[kind], time);

            intervals_[kind].push_back({timer_start, timer_stop});
        }
    }

    const intervals_list &intervals(int kind) { return intervals_[kind]; }

private:
    int depth_[max_kind] = {0};
    int tid_;

    time_point timer_start_[max_kind];

    long num_calls_[max_kind] = {0};
    static constexpr long skip_calls_ = 1;

    double min_time_[max_kind] = {DBL_MAX};
    double max_time_[max_kind] = {-DBL_MAX};
    double tot_time_[max_kind] = {0};

    intervals_list intervals_[max_kind];
};

struct ProfilingRegistry {
    ProfilingStats *register_stats() {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        registry_.push_back(new ProfilingStats(gettid()));
        return registry_.back();
    }
    ~ProfilingRegistry() {
        using namespace std::chrono;
        std::lock_guard<std::mutex> lock(registry_mutex_);
        if (registry_.size())
            printf("@@@ MKLDNN_NAIVE_PROFILER DATA v2\n");
        for (auto &ps: registry_)
        for (int kind = 0; kind < ProfilingStats::max_kind; kind++) {
            if (ps->num_calls(kind) == 0)
                continue;
            printf("@@@ MKLDNN_NAIVE_PROFILER: "
                    "%d %d %ld %10.2f %10.2f %10.2f",
                    ps->tid(), kind,
                    ps->num_calls(kind), ps->min_time(kind),
                    ps->avg_time(kind), ps->max_time(kind));
            for (auto &i: ps->intervals(kind)) {
                auto start = i.first.time_since_epoch();
                auto stop = i.second.time_since_epoch();
                printf(" %10.2f %10.2f",
                        duration<double, std::micro>(start).count(),
                        duration<double, std::micro>(stop).count());
            }
            printf("\n");
        }
    }

private:
    std::vector<ProfilingStats*> registry_;
    std::mutex registry_mutex_;
};

static ProfilingRegistry profiling_registry;

struct ThreadProfiler {
    ThreadProfiler(): stats_(profiling_registry.register_stats()) { }
    void start_timer(int kind) { stats_->start_timer(kind); }
    void stop_timer(int kind) { stats_->stop_timer(kind); }
private:
    ProfilingStats *stats_;
};

static thread_local ThreadProfiler thread_profiler;
void start(int kind) { thread_profiler.start_timer(kind); }
void stop(int kind) { thread_profiler.stop_timer(kind); }

#else
void start(int kind) {(void)kind;}
void stop(int kind) {(void)kind;}
#endif
}
}
}

