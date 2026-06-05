#pragma once
#include <functional>
#include <memory>
#include <atomic>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>

namespace xai {

class BarrierPool {
public:
    explicit BarrierPool(int n);
    ~BarrierPool();

    void parallel_for(int total, const std::function<void(int,int,int)> &func);
    int num_workers() const { return nthreads_; }

private:
    struct Range { int lo = 0, hi = 0; };

    void worker_loop(int id);

    std::vector<std::thread> workers_;
    std::vector<Range>       ranges_;
    const std::function<void(int,int,int)> *func_ = nullptr;

    // Синхронизация через condition variable вместо busy-wait
    std::mutex              mutex_;
    std::condition_variable cv_;
    std::condition_variable cv_done_;
    
    int  phase_     = 0;
    int  done_count_ = 0;
    bool stop_      = false;
    bool work_ready_ = false;

    int nthreads_;
    int active_ = 0;
};

extern std::unique_ptr<BarrierPool> g_pool;

} // namespace xai