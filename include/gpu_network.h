#pragma once

#include <vector>

#include "nccl.h"

class GPUNetwork
{
public:
    void AllGather();
    void Initialize(int buffer_size);
private:
    int n_dev;

    ncclComm_t comms;
    cudaStream_t* streams;
    char** buffers;
    char** host_buffers;
    cudaEvent_t timers;
};
