#pragma once

#include <vector>

#include "nccl.h"

class GPUNetwork {
public:
    void Initialize();
    
    void Point2Point();
    void AllGather();
    void GatherData();
    void Print();

    void Cleanup();
private:
    void ProfileCollective(void (GPUNetwork::* call)(void*), void* args);
    void P2PUniCall(void* args);
    void P2PBiCall(void* args);
    void AllGatherCall(void* args);
    void WriteTime(float* address, int scale);

    const int kTimingIters = 128;
    const int kWarmupIters = 8;
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

    float* aggregated_uni_times_;
    float* aggregated_bi_times_;
    float* aggregated_ag_times_;
};
