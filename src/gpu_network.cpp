#include <vector>

#include "nccl.h"

#include "error_guards.h"
#include "gpu_network.h"

void GPUNetwork::Initialize() {
    // Create NCCL comm
    CUDACHECK(cudaGetDeviceCount(&n_dev));
    int* dev_ids = new int[n_dev];
    for (int i = 0; i < n_dev; i++) {
        dev_ids[i] = i;
    }
    NCCLCHECK(ncclCommInitAll(&comms, n_dev, dev_ids));
    delete[] dev_ids;
}

void GPUNetwork::AllGather() {
    
}
