#include <stdlib.h>
#include "mpi.h"
#include "nccl.h"

#include "textor.h"

#include "gpu_network.h"
#include "error_guards.h"

void TExtor::Initialize(GPUNetwork* network) {
    net_ = network;

    sandwich_data_ = (swpmd_t*) malloc(
        net_->size_ * net_->size_ * net_->size_ * sizeof(swpmd_t)
    );
    if (sandwich_data_ == NULL) exit(-1);

    return;
}

void TExtor::Destroy() {
    free(sandwich_data_);
}

void TExtor::ProbeNetwork(int ppp) {
    cudaGraph_t sandwich_graph;
    cudaGraphExec_t sandwich;

    for (int iter = 0; iter < ppp; iter++) {
    for (int src = 0; src < net_->size_; src++) {
    for (int dst_a = 0; dst_a < net_->size_; dst_a++) {
    for (int dst_b = 0; dst_b < net_->size_; dst_b++) {
    if ((src != dst_a) && (src != dst_b) && (dst_a != dst_b)) {
        cudaStreamBeginCapture(net_->stream_, cudaStreamCaptureModeGlobal);
        NCCLCHECK(ncclGroupStart());
        if (net_->rank_ == src) {
            NCCLCHECK(ncclSend(net_->buffer_, net_->kBufferSize, ncclUint8, dst_b, net_->comm_, net_->stream_));
        } else if (net_->rank_ == dst_b) {
            NCCLCHECK(ncclRecv(net_->buffer_, net_->kBufferSize, ncclUint8, src, net_->comm_, net_->stream_));
        }
        NCCLCHECK(ncclGroupEnd());
        NCCLCHECK(ncclGroupStart());
        if (net_->rank_ == src) {
            CUDACHECK(cudaEventRecord(net_->stop_timer_, net_->stream_));
            NCCLCHECK(ncclSend(net_->buffer_, 1, ncclUint8, dst_a, net_->comm_, net_->stream_));
        } else if (net_->rank_ == dst_a) {
            CUDACHECK(cudaEventRecord(net_->stop_timer_, net_->stream_));
            NCCLCHECK(ncclRecv(net_->buffer_, 1, ncclUint8, src, net_->comm_, net_->stream_));
        }
        NCCLCHECK(ncclGroupEnd());
        cudaStreamEndCapture(net_->stream_, &sandwich_graph);
        cudaGraphInstantiate(&sandwich, sandwich_graph, NULL, NULL, 0);

        NCCLCHECK(ncclGroupStart());
        if (net_->rank_ == src) {
            NCCLCHECK(ncclSend(net_->buffer_, 1, ncclUint8, dst_a, net_->comm_, net_->stream_));
            CUDACHECK(cudaEventRecord(net_->start_timer_, net_->stream_));
            NCCLCHECK(ncclSend(net_->buffer_, 1, ncclUint8, dst_a, net_->comm_, net_->stream_));
        } else if (net_->rank_ == dst_a) {
            NCCLCHECK(ncclRecv(net_->buffer_, 1, ncclUint8, src, net_->comm_, net_->stream_));
            CUDACHECK(cudaEventRecord(net_->start_timer_, net_->stream_));
            NCCLCHECK(ncclRecv(net_->buffer_, 1, ncclUint8, src, net_->comm_, net_->stream_));
        }
        NCCLCHECK(ncclGroupEnd());
        cudaGraphLaunch(sandwich, net_->stream_);
        cudaStreamSynchronize(net_->stream_);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    }
    }
    }
    }

    return;
}

void TExtor::Characterize() {
    return;
}

void TExtor::Search() {
    return;
}

void TExtor::SingleProbe(int src, int dst_a, int dst_b) {
    return;
}
