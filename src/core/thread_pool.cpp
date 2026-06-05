#include "thread_pool.h"
#include <algorithm>

namespace xai {

std::unique_ptr<BarrierPool> g_pool;

BarrierPool::BarrierPool(int n) : nthreads_(n) {
    ranges_.resize(n);
    active_ = 0;
    func_   = nullptr;
    stop_   = false;
    work_ready_ = false;
    phase_  = 0;
    done_count_ = 0;

    for (int i = 0; i < n; i++) {
        workers_.emplace_back([this, i] { worker_loop(i); });
    }
}

BarrierPool::~BarrierPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
        work_ready_ = true;
    }
    cv_.notify_all();
    for (auto &w : workers_) {
        if (w.joinable()) w.join();
    }
}

void BarrierPool::parallel_for(int total,
                               const std::function<void(int,int,int)> &func) {
    if (total <= 0) return;
    if (nthreads_ == 0 || total < 2) {
        func(0, 0, total);
        return;
    }

    int nt    = std::min(nthreads_, total);
    int chunk = (total + nt - 1) / nt;
    for (int i = 0; i < nt; i++) {
        ranges_[i].lo = i * chunk;
        ranges_[i].hi = std::min(ranges_[i].lo + chunk, total);
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        active_     = nt;
        func_       = &func;
        done_count_ = 0;
        phase_++;
        work_ready_ = true;
    }
    cv_.notify_all();

    // Ждём завершения всех воркеров
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_done_.wait(lock, [this]() {
            return done_count_ >= active_;
        });
    }
}

void BarrierPool::worker_loop(int id) {
    int my_phase = 0;

    while (true) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            // Ждём новую работу или остановку
            cv_.wait(lock, [this, &my_phase]() {
                return stop_ || (work_ready_ && phase_ != my_phase);
            });

            if (stop_) return;

            // Обновляем фазу
            my_phase = phase_;
        }

        // Выполняем работу если наш id в диапазоне
        if (id < active_ && func_) {
            (*func_)(id, ranges_[id].lo, ranges_[id].hi);
        }

        // Отмечаем завершение
        {
            std::lock_guard<std::mutex> lock(mutex_);
            done_count_++;
            if (done_count_ >= active_) {
                work_ready_ = false;
                cv_done_.notify_one();
            }
        }
    }
}

} // namespace xai