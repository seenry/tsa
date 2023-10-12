#pragma once

#include <vector>

#include "nccl.h"

class GPUNetwork
{
public:
    void Initialize();
    
    void Point2Point(); // Measure all p2p connections
    void AllGather(); //
private:
    void Point2PointSingle(int from, int to);
    void Point2PointBidirectional(int rank_1, int rank_2);

    const int kTimingIters = 128;
    const int kBufferSize = 1 << 23;

    int n_ranks_;
    int rank_;

    ncclUniqueId id_;
    ncclComm_t comm_;
    cudaStream_t stream_;
    char* buffer_;
    char* host_buffer_;
    cudaEvent_t start_timer_;
    cudaEvent_t stop_timer_;

    float* p2p_uni_times_;
    float* p2p_bi_times_;
    float ag_time_;
};
