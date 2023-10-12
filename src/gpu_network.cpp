#include <vector>
#include <stdlib.h>
#include <stdio.h>

#include "nccl.h"
#include "mpi.h"

#include "error_guards.h"
#include "gpu_network.h"

void GPUNetwork::Initialize() {
    MPICHECK(MPI_Init(NULL, NULL));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank_))                     ;
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &n_ranks_))                  ;
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (rank_ == 0) {
        NCCLCHECK(ncclGetUniqueId(&id_))                                ;
    }
    MPI_Bcast(&id_, sizeof(id_), MPI_BYTE, rank_, MPI_COMM_WORLD)       ;
    NCCLCHECK(ncclCommInitRank(&comm_, n_ranks_, id_, rank_))           ;

    CUDACHECK(cudaSetDevice(rank_))                                     ;
    CUDACHECK(cudaMalloc(&buffer_, kBufferSize * sizeof(char)))         ;
    CUDACHECK(cudaMemset(&buffer_, rank_, kBufferSize * sizeof(char)))  ;
    CUDACHECK(cudaStreamCreate(&stream_))                               ;
    CUDACHECK(cudaEventCreate(&start_timer_))                           ;
    CUDACHECK(cudaEventCreate(&stop_timer_))                            ;

    p2p_uni_times_ = (float*) calloc(n_ranks_ * n_ranks_, sizeof(float));
    p2p_bi_times_ = (float*) calloc(n_ranks_ * n_ranks_, sizeof(float)) ;
    // ag_times_ = (float*) calloc(n_ranks_ * n_ranks_, sizeof(float))     ;
}

void GPUNetwork::Point2Point() {
    for (int i = 0; i < (n_ranks_ - 1); i++) {
        for (int j = 0; j < n_ranks_; j++) {
            Point2PointSingle(i, j)                                                                 ;
            if (rank_ == i || rank_ == j) {
                cudaEventElapsedTime(p2p_uni_times_ + (i * n_ranks_ + j), start_timer_, stop_timer_);
            }

            Point2PointSingle(j, i)                                                                 ;
            if (rank_ == i || rank_ == j) {
                cudaEventElapsedTime(p2p_uni_times_ + (j * n_ranks_ + i), start_timer_, stop_timer_);
            }

            MPI_Barrier(MPI_COMM_WORLD)                                                             ;
        }
    }

    
    for (int i = 0; i < (n_ranks_ - 1); i++) {
        for (int j = i + 1; j < n_ranks_; j++) {
            Point2PointBidirectional(i, j)                                                          ;
            if (rank_ == i || rank_ == j) {
                cudaEventElapsedTime(p2p_bi_times_ + (i * n_ranks_ + j), start_timer_, stop_timer_) ;
            }

            MPI_Barrier(MPI_COMM_WORLD)                                                             ;
        }
    }
}

void GPUNetwork::AllGather() {
    int msg_size = kBufferSize / n_ranks_                                                               ;
    NCCLCHECK(ncclAllGather(buffer + rank * 1, buffer, 1, ncclUint8, comm_, stream_))                   ;
    CUDACHECK(cudaEventRecord(start_timer_, stream_))                                                   ;
    for (int i = 0; i < kTimingIters; i++) {
        NCCLCHECK(ncclAllGather(buffer + rank * msg_size, buffer, msg_size, ncclUint8, comm_, stream_)) ;
    }
    CUDACHECK(cudaEventRecord(stop_timer_, stream_))                                                    ;
    CUDACHECK(cudaEventSynchronize(stop_timer_))                                                        ;
    cudaEventElapsedTime(&ag_time_, start_timer_, stop_timer_)                                          ;
}

void GPUNetwork::Print() {
    printf("Unidirectional P2P:");
    for (int i = 0; i < n_ranks_; i++) {
        for (int j = 0; j < n_ranks_; j++) {
            if (i != j) {
                printf("%6.2f ", p2p_uni_times_[i * n_ranks_ + j]);
            } else {
                printf("0.0000 ");
            }
        }
        printf("\n");
    }
    printf("\nBidirectional P2P:");
    for (int i = 0; i < (n_ranks_ - 1); i++) {
        for (int j = 0; j < i; j++) {
            printf("       ");
        }
        for (int j = i + 1; j < n_ranks_; j++) {
            printf("%6.2f ", p2p_bi_times_[i * n_ranks_ + j]);
        }
        printf("\n");
    }
    printf("\nAll Gather: %f", ag_time_);
}

void GPUNetwork::Point2PointSingle(int from, int to) {
    if (rank_ == from) {
        // Synchronizing sendrecv
        NCCLCHECK(ncclSend(buffer_, 1, ncclUint8, to, comm_, stream_))                  ;
        CUDACHECK(cudaEventRecord(start_timer_, stream_))                               ;
        // Should this be a NCCL group?
        // NCCL docs claim it lowers launch latency
        // Do I actually want to do that though?
        // i.e. does this mess up my latency profiling?
        NCCLCHECK(ncclGroupStart())                                                     ;
        for (int i = 0; i < kTimingIters; i++) {
            NCCLCHECK(ncclSend(buffer_, kBufferSize, ncclUint8, to, comm_, stream_))    ;
        }
        NCCLCHECK(ncclGroupEnd())                                                       ;
        CUDACHECK(cudaEventRecord(stop_timer_, stream_))                                ;
        CUDACHECK(cudaEventSynchronize(stop_timer_))                                    ;
    }
    else if (rank_ == to) {
        NCCLCHECK(ncclRecv(buffer_, 1, ncclUint8, from, comm_, stream_))                ;
        CUDACHECK(cudaEventRecord(start_timer_, stream_))                               ;
        NCCLCHECK(ncclGroupStart())                                                     ;
        for (int i = 0; i < kTimingIters; i++) {
            NCCLCHECK(ncclRecv(buffer_, kBufferSize, ncclUint8, from, comm_, stream_))  ;
        }
        NCCLCHECK(ncclGroupEnd())                                                       ;
        CUDACHECK(cudaEventRecord(stop_timer_, stream_))                                ;
        CUDACHECK(cudaEventSynchronize(stop_timer_))                                    ;
    }
}

void GPUNetwork::Point2PointBidirectional(int rank_1, int rank_2) {
    if (rank_ == rank_1) {
        NCCLCHECK(ncclSend(buffer_, 1, ncclUint8, rank_2, comm_, stream_))              ;
        CUDACHECK(cudaEventRecord(start_timer_, stream_))                               ;
        NCCLCHECK(ncclGroupStart())                                                     ;
        for (int i = 0; i < (kTimingIters >> 1); i++) {
            NCCLCHECK(ncclSend(buffer_, kBufferSize, ncclUint8, rank_2, comm_, stream_));
            NCCLCHECK(ncclRecv(buffer_, kBufferSize, ncclUint8, rank_2, comm_, stream_));
        }
        NCCLCHECK(ncclGroupEnd())                                                       ;
        CUDACHECK(cudaEventRecord(stop_timer_, stream_))                                ;
        CUDACHECK(cudaEventSynchronize(stop_timer_))                                    ;
    }
    else if (rank_ == rank_2) {
        NCCLCHECK(ncclRecv(buffer_, 1, ncclUint8, rank_1, comm_, stream_))              ;
        CUDACHECK(cudaEventRecord(start_timer_, stream_))                               ;
        NCCLCHECK(ncclGroupStart())                                                     ;
        for (int i = 0; i < (kTimingIters >> 1); i++) {
            NCCLCHECK(ncclRecv(buffer_, kBufferSize, ncclUint8, rank_1, comm_, stream_));
            NCCLCHECK(ncclSend(buffer_, kBufferSize, ncclUint8, rank_1, comm_, stream_));
        }
        NCCLCHECK(ncclGroupEnd())                                                       ;
        CUDACHECK(cudaEventRecord(stop_timer_, stream_))                                ;
        CUDACHECK(cudaEventSynchronize(stop_timer_))                                    ;
    }
    MPI_Barrier(MPI_COMM_WORLD)                                                         ;
}
