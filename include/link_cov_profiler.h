#pragma once

#include "profiler_interface.h"
#include "gpu_network.h"

class LinkCovProfiler : public IProfiler {
public:
    void Initialize(GPUNetwork* network);

    void ProfileOperation();
    void GatherResults();
    void PrintResults();
    
private:
    void OperationCall(int rank_1, int rank_2, int msg_size);
    void SingleCall(int rank_1, int rank_2, int msg_size);
    void GetVerticesFromEdge(int* v0_ptr, int* v1_ptr, int edge_index);

    int n_links_;
};
