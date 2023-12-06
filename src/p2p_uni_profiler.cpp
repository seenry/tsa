#include <stdlib.h>
#include <stdio.h>
#include <math.h>

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

    if (net_->rank_ == 0) {
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
        if (net_->rank_ == rank_1 || net_->rank_ == rank_2) {
            idx = rank_1 * net_->size_ + rank_2;
            fg = idx > (net_->rank_ * net_->size_ + net_->rank_);
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
        }
    }
    }
    }
};

void P2PUniProfiler::GatherResults() {
    for (int i = 0; i < n_iter_; i++) {
        NCCLCHECK(ncclReduce(
            (void*) (op_times_[i]), (void*) (output[i]),
            2 * net_->size_ * net_->size_,
            ncclFloat32, ncclMax,
            0, net_->comm_, net_->stream_
        ));
    }
}

void P2PUniProfiler::PrintResults() {
    for (int i = 0; i < net_->size_; i++) {
        if (net_->rank_ == i) {
            FILE* f;
            if (i != 0) {
                f = fopen("out.txt", "a");
            } else {
                f = fopen("out.txt", "w");
            }

            fprintf(f, "rank: %d\n", i);
            for (int j = 0; j < n_iter_; j++) {
                fprintf(f, "%d bytes:\n", 1 << j);
                for (int k = 0; k < (net_->size_ * 2); k++) {
                    for (int l = 0; l < net_->size_; l++) {
                        fprintf(f, "%f ", op_times_[j][k * net_->size_ + l]);
                    }
                    fprintf(f, "\n");
                }
                fprintf(f, "\n");
            }
            fclose(f);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (net_->rank_ == 0) {
        FILE* f = fopen("aggregated.txt", "w");
        for (int i = 0; i < n_iter_; i++) {
            fprintf(f, "%d bytes:\n", 1 << i);
            for (int j = 0; j < (net_->size_ * 2); j++) {
                for (int k = 0; k < net_->size_; k++) {
                    fprintf(f, "%f ", output[i][j * net_->size_ + k]);
                }
                fprintf(f, "\n");
            }
            fprintf(f, "\n\n");
        }
        fclose(f);
    }
}

void P2PUniProfiler::Cleanup() {
    for (int i = 0; i < n_iter_; i++) {
        free(op_times_[i]);
    }
    free(op_times_);

    if (net_->rank_ == 0) {
        for (int i = 0; i < n_iter_; i++) {
            free(output[i]);
        }
        free(output);
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
    if (net_->rank_ == rank_1) {
        NCCLCHECK(ncclSend(net_->buffer_, msg_size, ncclUint8, rank_2, net_->comm_, net_->stream_));
    } else if (net_->rank_ == rank_2) {
        NCCLCHECK(ncclRecv(net_->buffer_, msg_size, ncclUint8, rank_1, net_->comm_, net_->stream_));
    }
}
