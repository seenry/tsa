#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>

#include "mpi.h"
#include "nccl.h"

#include "profiler_interface.h"
#include "gpu_network.h"

#include "error_guards.h"

class P2PUniProfiler : public IProfiler {
public:
    void Initialize(GPUNetwork* network);

    void ProfileOperation();
    void GatherResults();
    void PrintResults();
    
    void Cleanup();
    
private:
    void OperationCall(int rank_1, int rank_2, int msg_size);
    void SingleCall(int rank_1, int rank_2, int msg_size);
};
