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

    int n_links_;
};
