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

    DetermineRank();

    if (rank_ == 0) {
        NCCLCHECK(ncclGetUniqueId(&id_));
    }
    MPICHECK(MPI_Bcast((void*) &id_, sizeof(id_), MPI_BYTE, 0, mpi_comm_));

    CUDACHECK(cudaMalloc(&buffer_, kBufferSize * sizeof(char)));
    // CUDACHECK(cudaMemset(buffer_, rank_, kBufferSize * sizeof(char)));
    CUDACHECK(cudaStreamCreate(&stream_));
    CUDACHECK(cudaEventCreate(&start_timer_));
    CUDACHECK(cudaEventCreate(&stop_timer_));

    NCCLCHECK(ncclCommInitRank(&nccl_comm_, size_, id_, rank_));
}

void GPUNetwork::Cleanup() {
    CUDACHECK(cudaFree(buffer_));
    CUDACHECK(cudaStreamDestroy(stream_));
    CUDACHECK(cudaEventDestroy(start_timer_));
    CUDACHECK(cudaEventDestroy(stop_timer_));

    NCCLCHECK(ncclCommDestroy(nccl_comm_));
}

void GPUNetwork::DetermineRank() {
    int global_size, global_rank;
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &global_size));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &global_rank));

    // Find rank within node
    int hosthashes[global_size];
    char hostname[1024];
    gethostname(hostname, 1024);
    for (int i = 0; i < 1024; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            break;
        }
    }
    int hosthash = 5381;
    for (int i = 0; hostname[i] != '\0'; i++) {
        hosthash = ((hosthash << 5) + hosthash) ^ hostname[i];
    }
    hosthashes[global_rank] = hosthash & 0x7fffffff;
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hosthashes, sizeof(int), MPI_BYTE, MPI_COMM_WORLD));

    int local_rank = 0;
    for (int i = 0; i < global_size; i++) {
        if (i == global_rank) break;
        if (hosthashes[i] == hosthashes[global_rank]) local_rank++;
    }
    
    MPI_Comm node_comm, leader_comm;
    // Create communicator for each node
    MPICHECK(MPI_Comm_split(MPI_COMM_WORLD, hosthash, local_rank, &node_comm));
    // Create communicator between nodes
    MPICHECK(MPI_Comm_split(MPI_COMM_WORLD, local_rank == 0, global_rank, &leader_comm));

    // Get a logical ID for each node
    int node_id;
    MPICHECK(MPI_Comm_rank(leader_comm, &node_id));
    MPICHECK(MPI_Bcast((void*) &node_id, sizeof(node_id), MPI_BYTE, 0, node_comm));

    // Assign GPU to rank (assume nodes are balanced)
    int node_size;
    MPICHECK(MPI_Comm_size(node_comm, &node_size));
    rank_ = node_id * (node_size >> 1) + (local_rank >> 1);
    CUDACHECK(cudaSetDevice(local_rank >> 1));

    // Get two identical communicators
    // GPU on rank_i in one mpi_comm_ should correspond to same GPU on rank_i in other mpi_comm_
    MPICHECK(MPI_Comm_split(MPI_COMM_WORLD, local_rank % 2, rank_, &mpi_comm_));
    MPICHECK(MPI_Comm_size(mpi_comm_, &size_));

    comm_rank_ = local_rank;
    MPICHECK(MPI_Bcast((void*) &comm_rank_, sizeof(comm_rank_), MPI_BYTE, 0, mpi_comm_));

    printf("Process %d on %s (%d). GPU %d/%d for comm %d\n", global_rank, hostname, node_id, rank_, size_ - 1, comm_rank_);

    MPICHECK(MPI_Comm_free(&node_comm));
    MPICHECK(MPI_Comm_free(&leader_comm));
}
