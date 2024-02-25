#pragma once

#include <stdlib.h>

#include "gpu_network.h"
#include "error_guards.h"

class IProfiler {
public:
    virtual void Initialize(GPUNetwork* network) = 0;

    virtual void ProfileOperation() = 0;
    virtual void GatherResults() = 0;
    virtual void PrintResults() = 0;

    virtual void Cleanup() {
        for (int i = 0; i < n_iter_; i++) {
            free(op_times_[i]);
        }
        free(op_times_);

        if (net_->rank_ == 0) {
            for (int i = 0; i < n_iter_; i++) {
                free(output[i]);
            }
            free(output);
        }
    };
    
protected:
    void RecordTime(float* address, int scale) {
        CUDACHECK(cudaEventElapsedTime(address, net_->start_timer_, net_->stop_timer_));
        *address /= (float) scale;
    };

    const int kTimingIters = 128;
    const int kWarmupIters = 4;

    GPUNetwork* net_;

    float** op_times_;
    float** output;
    int n_iter_;
};
