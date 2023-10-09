#include <vector>

#include "nccl.h"

#include "error_guards.h"
#include "gpu_network.h"

void GPUNetwork::Initialize(int buffer_size) {
    CUDACHECK(cudaGetDeviceCount(&n_gpus));
    int* dev_ids = new int[n_gpus];
    for (int i = 0; i < n_gpus; i++) {
        dev_ids[i] = i;
    }

    streams = ;
    buffers = ;
    host_buffers = ;
    

    CUDACHECK(cudaSetDevice(MY_RANK));
    CUDACHECK(cudaMalloc(&buffer, buffer_size * sizeof(char)));
    CUDACHECK(cudaMemset(&buffer, i, buffer_size * sizeof(char)));
    CUDACHECK(cudaStreamCreate(streams + i));

    NCCLCHECK(ncclCommInitAll(&comms, n_gpus, dev_ids));
    delete[] dev_ids;
}

void GPUNetwork::AllGather() {
    
}
