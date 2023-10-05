#pragma once

#include <vector>

#include "nccl.h"

class GPUNetwork
{
public:
    void AllGather();
    void Initialize();
private:
    std::vector<ncclComm_t> comms;
    std::vector<cudaStream_t> streams;
    std::vector<std::vector<char>> buffers;
    std::vector<std::vector<char>> host_buffers;
    std::vector<cudaEvent_t> timers;

}