#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>

#include "nccl.h"
#include "mpi.h"

#include "error_guards.h"
#include "gpu_network.h"

void GPUNetwork::Initialize() {
    MPICHECK(MPI_Init(NULL, NULL));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &node_rank_));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &size_));

    SetGPU();

    if (node_rank_ == 0) {
        NCCLCHECK(ncclGetUniqueId(&id_));
    }
    MPICHECK(MPI_Bcast((void*) &id_, sizeof(id_), MPI_BYTE, 0, MPI_COMM_WORLD));

    CUDACHECK(cudaMalloc(&buffer_, kBufferSize * sizeof(char)));
    // CUDACHECK(cudaMemset(buffer_, node_rank_, kBufferSize * sizeof(char)));
    CUDACHECK(cudaStreamCreate(&stream_));
    CUDACHECK(cudaEventCreate(&start_timer_));
    CUDACHECK(cudaEventCreate(&stop_timer_));

    NCCLCHECK(ncclCommInitRank(&comm_, size_, id_, node_rank_));
}

void GPUNetwork::Cleanup() {
    CUDACHECK(cudaFree(buffer_));
    CUDACHECK(cudaStreamDestroy(stream_));
    CUDACHECK(cudaEventDestroy(start_timer_));
    CUDACHECK(cudaEventDestroy(stop_timer_));

    NCCLCHECK(ncclCommDestroy(comm_));
}

void GPUNetwork::SetGPU() {
    uint64_t hosthashes[size_];
    char hostname[1024];
    gethostname(hostname, 1024);
    for (int i = 0; i < 1024; i++) {
		if (hostname[i] == '.') {
			hostname[i] = '\0';
			break;
		}
	}
	uint64_t hosthash = 5381;
	for (int i = 0; hostname[i] != '\0'; i++) {
		hosthash = ((hosthash << 5) + hosthash) ^ hostname[i];
	}
	hosthashes[node_rank_] = hosthash;
	MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hosthashes, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));

	local_rank_ = 0;
	for (int i = 0; i < size_; i++) {
		if (i == node_rank_) break;
		if (hosthashes[i] == hosthashes[node_rank_]) local_rank_++;
	}

    CUDACHECK(cudaSetDevice(local_rank_));

    printf("gpu %d on node %d\n", local_rank_, node_rank_);
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
}
