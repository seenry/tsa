#pragma once

#include "nccl.h"

class GPUNetwork
{
public:
    void Initialize(int buffer_size);
    
    void Point2Point(); // Measure all p2p connections
    void AllGather(); //
private:
    void Point2PointSingle(int from, int to);
    void Point2PointBidirectional(int rank_1, int rank_2);

    int n_gpus;
    int my_rank;

    ncclComm_t comm;
    cudaStream_t stream;
    char* buffer;
    char* host_buffer;
    cudaEvent_t start_timer;
    cudaEvent_t stop_timer;
};
