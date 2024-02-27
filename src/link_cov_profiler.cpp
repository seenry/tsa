#include <stdlib.h>
#include <stdio.h>
#include <cmath>

#include "mpi.h"
#include "nccl.h"

#include "link_cov_profiler.h"
#include "error_guards.h"

void LinkCovProfiler::Initialize(GPUNetwork* network) {
    net_ = network;

    n_msg_ = 0;
    for (int i = 1; i <= net_->kBufferSize; i <<= 1) {
        n_msg_++;
    }

    n_links_ = net_->size_ * (net_->size_ - 1) * 0.5;

    op_times_ = (float**) malloc(n_msg_ * sizeof(float*));
    CheckAlloc(op_times_, "link covariance operation times array");
    for (int i = 0; i < n_msg_; i++) {
        op_times_[i] = (float*) calloc(2 * n_links_ * n_links_, sizeof(float));
        CheckAlloc(op_times_[i], "link covariance operation time matrix");
    }

    output = (float**) malloc(n_msg_ * sizeof(float*));
    CheckAlloc(output, "link covariance outputs");
    for (int i = 0; i < n_msg_; i++) {
        output[i] = (float*) calloc(2 * net_->size_ * net_->size_, sizeof(float));
        CheckAlloc(output[i], "link covariance output");
    }
}

void LinkCovProfiler::ProfileOperation() {
    int idx, fg;
    
    int src_0, dst_0, src_1, dst_1;
    for (int msg = 0; msg < n_msg_; msg++) {
    for (int round = 0; round < kNRounds; round++) {
    for (int link_0 = 0; link_0 < (n_links_ - 1); link_0++) {
    for (int link_1 = link_0 + 1; link_1 < n_links_; link_1++) {
        GetVerticesFromEdge(&src_0, &dst_0, link_0);
        GetVerticesFromEdge(&src_1, &dst_1, link_1);
        // if (net_->rank_ == rank_1 || net_->rank_ == rank_2) {
        //     // idx = rank_1 * net_->size_ + rank_2;
        //     // fg = idx < (net_->rank_ * net_->size_ + net_->rank_);
        //     // OperationCall(rank_1, rank_2, 1 << msg);
        //     // RecordTime(
        //     //     op_times_[msg] + fg * net_->size_ * net_->size_ + idx,
        //     //     kTimingIters
        //     // );

        //     // idx = rank_2 * net_->size_ + rank_1;
        //     // OperationCall(rank_2, rank_1, 1 << msg);
        //     // RecordTime(
        //     //     op_times_[msg] + fg * net_->size_ * net_->size_ + idx,
        //     //     kTimingIters
        //     // );
        // }
    }}}}
};

void LinkCovProfiler::GatherResults() {
    for (int i = 0; i < n_msg_; i++) {
        MPICHECK(MPI_Reduce(
            op_times_[i], output[i],
            2 * net_->size_ * net_->size_,
            MPI_FLOAT, MPI_SUM,
            0, MPI_COMM_WORLD
        ));
    }
}

void LinkCovProfiler::PrintResults() {
    if (net_->rank_ == 0) {
        FILE* f = fopen("p2p_uni.csv", "w");
        fprintf(f, "n_bytes, from, to, min, max, avg\n");
        for (int i = 0; i < n_msg_; i++) {
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

void LinkCovProfiler::OperationCall(int rank_1, int rank_2, int msg_size) {
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

void LinkCovProfiler::SingleCall(int rank_1, int rank_2, int msg_size) {
    if (net_->rank_ == rank_1) {
        NCCLCHECK(ncclSend(net_->buffer_, msg_size, ncclUint8, rank_2, net_->nccl_comm_, net_->stream_));
    } else if (net_->rank_ == rank_2) {
        NCCLCHECK(ncclRecv(net_->buffer_, msg_size, ncclUint8, rank_1, net_->nccl_comm_, net_->stream_));
    }
}

void LinkCovProfiler::GetVerticesFromEdge(int* v0_ptr, int* v1_ptr, int edge_index) {
    int v1 = floor((sqrt(8 * edge_index + 1) - 1) * .5) + 1;
    *v0_ptr = edge_index - (v1 * (v1 - 1)) * .5;
    *v1_ptr = v1;
}
