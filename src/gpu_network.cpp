#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "nccl.h"
#include "mpi.h"

#include "error_guards.h"
#include "gpu_network.h"

///////////////////////////////////////////////////////////////////////////////
/* Public Function Definitions                                               */
///////////////////////////////////////////////////////////////////////////////
void GPUNetwork::Initialize() {
    MPICHECK(MPI_Init(NULL, NULL));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank_));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &n_ranks_));
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (rank_ == 0) {
        NCCLCHECK(ncclGetUniqueId(&id_));
    }
    MPI_Bcast((void*) &id_, sizeof(id_), MPI_BYTE, 0, MPI_COMM_WORLD);

    CUDACHECK(cudaSetDevice(rank_));
    CUDACHECK(cudaMalloc(&buffer_, kBufferSize * sizeof(char)));
    CUDACHECK(cudaMemset(buffer_, rank_, kBufferSize * sizeof(char)));
    CUDACHECK(cudaStreamCreate(&stream_));
    CUDACHECK(cudaEventCreate(&start_timer_));
    CUDACHECK(cudaEventCreate(&stop_timer_));

    NCCLCHECK(ncclCommInitRank(&comm_, n_ranks_, id_, rank_));

    p2p_uni_times_ = (float*) calloc(n_ranks_ * n_ranks_, sizeof(float));
    p2p_bi_times_ = (float*) calloc(n_ranks_ * n_ranks_, sizeof(float));

    if (rank_ == 0) {
        aggregated_uni_times_ = (float*) calloc(n_ranks_ * n_ranks_ * n_ranks_, sizeof(float));
        aggregated_bi_times_ = (float*) calloc(n_ranks_ * n_ranks_ * n_ranks_, sizeof(float));
        aggregated_ag_times_ = (float*) calloc(n_ranks_, sizeof(float));
    }
}

void GPUNetwork::Point2Point() {
    int endpoints[2] = {0, 0};
    for (int i = 0; i < (n_ranks_ - 1); i++) {
        for (int j = i + 1; j < n_ranks_; j++) {
            if (rank_ == i || rank_ == j) {
                endpoints[0] = i;
                endpoints[1] = j;
                ProfileCollective(&GPUNetwork::P2PUniCall, endpoints);
                WriteTime(p2p_uni_times_ + (i * n_ranks_ + j), kTimingIters);

                ProfileCollective(&GPUNetwork::P2PUniCall, endpoints);
                WriteTime(p2p_uni_times_ + (j * n_ranks_ + i), kTimingIters);

                ProfileCollective(&GPUNetwork::P2PBiCall, endpoints);
                WriteTime(p2p_bi_times_ + (i * n_ranks_ + j), kTimingIters * 2);
            }
        }
    }
}

void GPUNetwork::AllGather() {
    ProfileCollective(&GPUNetwork::AllGatherCall, NULL);
    for (int i = 0; i < n_ranks_; i++) {
        WriteTime(&ag_time_, kTimingIters);
    }
}

void GPUNetwork::GatherData() {
    MPI_Gather(p2p_uni_times_, n_ranks_ * n_ranks_, MPI_FLOAT, aggregated_uni_times_, n_ranks_ * n_ranks_, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(p2p_bi_times_, n_ranks_ * n_ranks_, MPI_FLOAT, aggregated_bi_times_, n_ranks_ * n_ranks_, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(&ag_time_, 1, MPI_FLOAT, aggregated_ag_times_, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

void GPUNetwork::Print() {
    if (rank_ == 0) {
        FILE* file_ptr;
        file_ptr = fopen("./p2p_uni.csv", "w");
        fprintf(file_ptr, "from,to,msg_size,average,median,min,max\n");
        for (int i = 0; i < (n_ranks_ - 1); i++) {
            for (int j = i + 1; j < n_ranks_; j++) {
                fprintf(file_ptr, "%d,%d,%d,%f,%f,%f,%f\n",
                        i, j, kBufferSize,
                        (aggregated_uni_times_[i * n_ranks_ * n_ranks_ + i * n_ranks_ + j] +
                         aggregated_uni_times_[j * n_ranks_ * n_ranks_ + i * n_ranks_ + j]) / 2,
                        (aggregated_uni_times_[i * n_ranks_ * n_ranks_ + i * n_ranks_ + j] +
                         aggregated_uni_times_[j * n_ranks_ * n_ranks_ + i * n_ranks_ + j]) / 2,
                        fmin(aggregated_uni_times_[i * n_ranks_ * n_ranks_ + i * n_ranks_ + j],
                             aggregated_uni_times_[j * n_ranks_ * n_ranks_ + i * n_ranks_ + j]),
                        fmax(aggregated_uni_times_[i * n_ranks_ * n_ranks_ + i * n_ranks_ + j],
                         aggregated_uni_times_[j * n_ranks_ * n_ranks_ + i * n_ranks_ + j])
                );
            }
        }
        fclose(file_ptr);
        
        file_ptr = fopen("./p2p_bi.csv", "w");
        fprintf(file_ptr, "from,to,msg_size,average,median,min,max\n");
        for (int i = 0; i < (n_ranks_ - 1); i++) {
            for (int j = i + 1; j < n_ranks_; j++) {
                fprintf(file_ptr, "%d,%d,%d,%f,%f,%f,%f\n",
                        i, j, kBufferSize,
                        (aggregated_bi_times_[i * n_ranks_ * n_ranks_ + i * n_ranks_ + j] +
                         aggregated_bi_times_[j * n_ranks_ * n_ranks_ + i * n_ranks_ + j]) / 2,
                        (aggregated_bi_times_[i * n_ranks_ * n_ranks_ + i * n_ranks_ + j] +
                         aggregated_bi_times_[j * n_ranks_ * n_ranks_ + i * n_ranks_ + j]) / 2,
                        fmin(aggregated_bi_times_[i * n_ranks_ * n_ranks_ + i * n_ranks_ + j],
                             aggregated_bi_times_[j * n_ranks_ * n_ranks_ + i * n_ranks_ + j]),
                        fmax(aggregated_bi_times_[i * n_ranks_ * n_ranks_ + i * n_ranks_ + j],
                         aggregated_bi_times_[j * n_ranks_ * n_ranks_ + i * n_ranks_ + j])
                );
            }
        }
        fclose(file_ptr);

        file_ptr = fopen("./all_gather.csv", "w");
        fprintf(file_ptr, "from,to,msg_size,average,median,min,max\n");
        std::sort(aggregated_ag_times_, aggregated_ag_times_ + n_ranks_);
        float mean = 0;
        float median = aggregated_ag_times_[n_ranks_ / 2];
        float min_ag = 100000000;
        float max_ag = 0;
        for (int i = 0; i < n_ranks_; i++) {
            mean += aggregated_ag_times_[i];
            if (aggregated_ag_times_[i] < min_ag) {
                min_ag = aggregated_ag_times_[i];
            }
            if (aggregated_ag_times_[i] > max_ag) {
                max_ag = aggregated_ag_times_[i];
            }
        }
        mean /= (float) n_ranks_;
        fprintf(file_ptr, "%d,%d,%d,%f,%f,%f,%f\n",
                -1, -1, kBufferSize,
                mean, median, min_ag, max_ag
        );
        fclose(file_ptr);
    }
}

void GPUNetwork::Cleanup() {
    CUDACHECK(cudaFree(buffer_));
    CUDACHECK(cudaStreamDestroy(stream_));
    CUDACHECK(cudaEventDestroy(start_timer_));
    CUDACHECK(cudaEventDestroy(stop_timer_));

    free(p2p_uni_times_);
    free(p2p_bi_times_);

    ncclCommDestroy(comm_);
    MPI_Finalize();
}

///////////////////////////////////////////////////////////////////////////////
/* Private Function Definitions                                              */
///////////////////////////////////////////////////////////////////////////////
void GPUNetwork::ProfileCollective(void (GPUNetwork::* call)(void*), void* args) {
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < kWarmupIters; i++) {
        (this->*call)(args);
    }
    NCCLCHECK(ncclGroupEnd());
    CUDACHECK(cudaEventRecord(start_timer_));
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < kTimingIters; i++) {
        (this->*call)(args);
    }
    NCCLCHECK(ncclGroupEnd());
    CUDACHECK(cudaEventRecord(stop_timer_));
    CUDACHECK(cudaEventSynchronize(stop_timer_));
}

void GPUNetwork::P2PUniCall(void* args) {
    int from = ((int*) args)[0];
    int to = ((int*) args)[1];
    if (rank_ == from) {
        NCCLCHECK(ncclSend(buffer_, kBufferSize, ncclUint8, to, comm_, stream_));
    } else if (rank_ == to) {
        NCCLCHECK(ncclRecv(buffer_, kBufferSize, ncclUint8, from, comm_, stream_));
    }
}

void GPUNetwork::P2PBiCall(void* args) {
    int rank_1 = ((int*) args)[0];
    int rank_2 = ((int*) args)[1];
    if (rank_ == rank_1) {
        NCCLCHECK(ncclSend(buffer_, kBufferSize, ncclUint8, rank_2, comm_, stream_));
        NCCLCHECK(ncclRecv(buffer_, kBufferSize, ncclUint8, rank_2, comm_, stream_));
    } else if (rank_ == rank_2) {
        NCCLCHECK(ncclRecv(buffer_, kBufferSize, ncclUint8, rank_1, comm_, stream_));
        NCCLCHECK(ncclSend(buffer_, kBufferSize, ncclUint8, rank_1, comm_, stream_));
    }
}

void GPUNetwork::AllGatherCall(void* args) {
    NCCLCHECK(ncclAllGather(
        buffer_ + rank_ * (kBufferSize >> 2),
        buffer_,
        kBufferSize >> 2,
        ncclUint8,
        comm_,
        stream_
    ));
}

void GPUNetwork::WriteTime(float* address, int scale) {
    CUDACHECK(cudaEventElapsedTime(address, start_timer_, stop_timer_));
    *address /= (float) scale;
}
