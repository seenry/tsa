#pragma once

#include "nccl.h"

class GPUNetwork {
public:
    void Initialize();
    void Cleanup();

    const int kBufferSize = 1 << 23;

    int comm_rank_;
    int size_;
    int rank_;

    MPI_Comm mpi_comm_;

    ncclUniqueId id_;
    ncclComm_t nccl_comm_;
    cudaStream_t stream_;
    char* buffer_;
    char* host_buffer_;
    cudaEvent_t start_timer_;
    cudaEvent_t stop_timer_;

private:
    void SetUpComms();
};
