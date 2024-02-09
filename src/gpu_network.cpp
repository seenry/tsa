#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "nccl.h"
#include "mpi.h"

#include "error_guards.h"
#include "gpu_network.h"

void GPUNetwork::Initialize() {
    MPICHECK(MPI_Init(NULL, NULL));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank_));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &size_));
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (rank_ == 0) {
        NCCLCHECK(ncclGetUniqueId(&id_));
    }
    MPICHECK(MPI_Bcast((void*) &id_, sizeof(id_), MPI_BYTE, 0, MPI_COMM_WORLD));

    CUDACHECK(cudaSetDevice(rank_));
    CUDACHECK(cudaMalloc(&buffer_, kBufferSize * sizeof(char)));
    // CUDACHECK(cudaMemset(buffer_, rank_, kBufferSize * sizeof(char)));
    CUDACHECK(cudaStreamCreate(&stream_));
    CUDACHECK(cudaEventCreate(&start_timer_));
    CUDACHECK(cudaEventCreate(&stop_timer_));

    NCCLCHECK(ncclCommInitRank(&comm_, size_, id_, rank_));
}

void GPUNetwork::Destroy() {
    CUDACHECK(cudaFree(buffer_));
    CUDACHECK(cudaStreamDestroy(stream_));
    CUDACHECK(cudaEventDestroy(start_timer_));
    CUDACHECK(cudaEventDestroy(stop_timer_));

    NCCLCHECK(ncclCommDestroy(comm_));
}
