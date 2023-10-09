#include <vector>

#include "nccl.h"

#include "error_guards.h"
#include "gpu_network.h"

void GPUNetwork::Initialize(int buffer_size) {
    CUDACHECK(cudaGetDeviceCount(&n_dev));
    int* dev_ids = new int[n_dev];
    for (int i = 0; i < n_dev; i++) {
        dev_ids[i] = i;
    }

    streams = new cudaStream_t[n_dev];
    buffers = new std::vector<char>[n_dev];
    host_buffers = new std::vector<char>
    for (int i = 0; i < n_dev; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc(buffers + i, buffer_size * sizeof(char)));
        CUDACHECK(cudaMemset(buffers[i], i, buffer_size * sizeof(char)));
        CUDACHECK(cudaStreamCreate(streams + i));
    }

    NCCLCHECK(ncclCommInitAll(&comms, n_dev, dev_ids));
    delete[] dev_ids;
}

void GPUNetwork::AllGather() {
    
}
