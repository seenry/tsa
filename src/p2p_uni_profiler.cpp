#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "p2p_uni_profiler.h"
#include "error_guards.h"

P2PUniProfiler::ProfileOperation() {
    for (int iter = 0; iter <= n_iter_; iter++) {
    for (int rank_1 = 0; rank_1 < (net_.size_ - 1); rank_1++) {
    for (int rank_2 = rank_1 + 1; rank_2 < net_.size_; rank_2++) {
        if (net_.rank_ == rank_1 || net_.rank_ == rank_2) {
            OperationCall(rank_1, rank_2, 1 << iter);
            RecordTime(
                op_times_ + (iter * net_.size_ * net_.size_) + (rank_1 * net_.size_) + rank_2,
                kTimingIters
            );

            OperationCall(rank_2, rank_1, 1 << iter);
            RecordTime(
                op_times_ + (iter * net_.size_ * net_.size_) + (rank_1 * net_.size_) + rank_2,
                kTimingIters
            );
        }
    }
    }
    }
}

P2PUniProfiler::PrintResults() {
    FILE* f_ptr;
}

P2PUniProfiler::OperationCall(int rank_1, int rank_2, int msg_size) {
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < kWarmupIters; i++) {
        SingleCall(rank_1, rank_2, 1);
    }
    NCCLCHECK(ncclGroupEnd());
    CUDACHECK(cudaEventRecord(net_.start_timer_));
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < kWarmupIters; i++) {
        SingleCall(rank_1, rank_2, msg_size);
    }
    NCCLCHECK(ncclGroupEnd());
    CUDACHECK(cudaEventRecord(net_.stop_timer_));
    CUDACHECK(cudaEventSynchronize(net_.stop_timer_));
}

P2PUniProfiler::SingleCall(int rank_1, int rank_2, ing msg_size) {
    if (net_.rank_ == rank_1) {
        NCCLCHECK(ncclSend(net_.buffer_, msg_size, ncclUint8, rank_2, net_.comm_, net_.stream_));
    } else if (net_.rank_ == rank_2) {
        NCCLCHECK(ncclRecv(net_.buffer_, msg_size, ncclUint8, rank_1, net_.comm_, net_.stream_));
    }
}
