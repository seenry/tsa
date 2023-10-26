#pragma once

#include <stdlib.h>

#include "gpu_network.h"
#include "error_guards.h"

class IProfiler {
public:
    void Initialize(GPUNetwork &network) {
        net_ = network;

        // since we double message size each iteration
        int n_iter_ = log2(net_.kBufferSize) + 1;

        op_times_ = (float*) calloc(
            n_iter * net_.size_ * net_.size_,
            sizeof(float)
        );
    };

    virtual void ProfileOperation() = 0;
    virtual void GatherResults() = 0;
    virtual void PrintResults() = 0;

    void Cleanup() {
        free(op_times_);
    };

private:
    void RecordTime(float* address, int scale) {
        CUDACHECK(cudaEventElapsedTime(address, net_.start_timer_, net_.stop_timer_));
        *address /= (float) scale;
    };

    const int kTimingIters = 128;
    const int kWarmupIters = 4;

    GPUNetwork net_;

    float* op_times_;
    int n_iter_;
}
