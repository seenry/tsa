#pragma once

#include "nccl.h"

class GPUNetwork {
public:
    void Initialize();
    void Cleanup();

    const int kBufferSize = 1 << 23;

    int size_;
    int rank_;
    int local_rank_;

    ncclUniqueId ids_[2];
    ncclComm_t comms_[2];
    cudaStream_t streams_[2];
    char* buffer_;
    char* host_buffer_;
    cudaEvent_t start_timer_;
    cudaEvent_t stop_timer_;

private:
    void GetLocalRank();
};
