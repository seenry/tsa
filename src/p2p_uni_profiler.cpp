#include <stdlib.h>
#include <stdio.h>
#include <cmath>

#include "nccl.h"

#include "p2p_uni_profiler.h"
#include "error_guards.h"

void P2PUniProfiler::Initialize(GPUNetwork* network) {
    net_ = network;

    n_iter_ = 0;
    for (int i = 1; i <= net_->kBufferSize; i <<= 1) {
        n_iter_++;
    }

    op_times_ = (float**) malloc(n_iter_ * sizeof(float*));
    for (int i = 0; i < n_iter_; i++) {
        op_times_[i] = (float*) calloc(2 * net_->size_ * net_->size_, sizeof(float));
    }

    if (net_->node_rank_ == 0) {
        output = (float**) malloc(n_iter_ * sizeof(float*));
        for (int i = 0; i < n_iter_; i++) {
            output[i] = (float*) calloc(2 * net_->size_ * net_->size_, sizeof(float));
        }
    }
}

void P2PUniProfiler::ProfileOperation() {
    int idx, fg;
    
    for (int iter = 0; iter < n_iter_; iter++) {
    for (int rank_1 = 0; rank_1 < (net_->size_ - 1); rank_1++) {
    for (int rank_2 = rank_1 + 1; rank_2 < net_->size_; rank_2++) {
    if (net_->node_rank_ == rank_1 || net_->node_rank_ == rank_2) {
        idx = rank_1 * net_->size_ + rank_2;
        fg = idx < (net_->node_rank_ * net_->size_ + net_->node_rank_);
        OperationCall(rank_1, rank_2, 1 << iter);
        RecordTime(
            op_times_[iter] + fg * net_->size_ * net_->size_ + idx,
            kTimingIters
        );

        idx = rank_2 * net_->size_ + rank_1;
        OperationCall(rank_2, rank_1, 1 << iter);
        RecordTime(
            op_times_[iter] + fg * net_->size_ * net_->size_ + idx,
            kTimingIters
        );
    }}}}
};

void P2PUniProfiler::GatherResults() {
    for (int i = 0; i < n_iter_; i++) {
        MPICHECK(MPI_Reduce(
            op_times_[i], output[i],
            2 * net_->size_ * net_->size_,
            MPI_FLOAT, MPI_SUM,
            0, MPI_COMM_WORLD
        ));
    }
}

void P2PUniProfiler::PrintResults() {
    if (net_->node_rank_ == 0) {
        FILE* f = fopen("p2p_uni.csv", "w");
        fprintf(f, "n_bytes, from, to, min, max, avg\n");
        for (int i = 0; i < n_iter_; i++) {
        for (int j = 0; j < net_->size_; j++) {
        for (int k = 0; k < net_->size_; k++) {
        if (j != k) {
            fprintf(f, "%d, %d, %d, %f, %f, %f\n",
                1 << i, j, k,
                fmin(output[i][j * net_->size_ + k], output[i][net_->size_ * net_->size_ + j * net_->size_ + k]),
                fmax(output[i][j * net_->size_ + k], output[i][net_->size_ * net_->size_ + j * net_->size_ + k]),
                (output[i][j * net_->size_ + k] + output[i][net_->size_ * net_->size_ + j * net_->size_ + k]) / 2
            );
        }}}}
        fclose(f);
    }
}

void P2PUniProfiler::OperationCall(int rank_1, int rank_2, int msg_size) {
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < kWarmupIters; i++) {
        SingleCall(rank_1, rank_2, 1);
    }
    NCCLCHECK(ncclGroupEnd());
    CUDACHECK(cudaEventRecord(net_->start_timer_));
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < kWarmupIters; i++) {
        SingleCall(rank_1, rank_2, msg_size);
    }
    NCCLCHECK(ncclGroupEnd());
    CUDACHECK(cudaEventRecord(net_->stop_timer_));
    CUDACHECK(cudaEventSynchronize(net_->stop_timer_));
}

void P2PUniProfiler::SingleCall(int rank_1, int rank_2, int msg_size) {
    if (net_->node_rank_ == rank_1) {
        NCCLCHECK(ncclSend(net_->buffer_, msg_size, ncclUint8, rank_2, net_->comm_, net_->stream_));
    } else if (net_->node_rank_ == rank_2) {
        NCCLCHECK(ncclRecv(net_->buffer_, msg_size, ncclUint8, rank_1, net_->comm_, net_->stream_));
    }
}
