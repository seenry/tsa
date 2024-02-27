#include <stdio.h>

#include "mpi.h"
#include "nccl.h"

#include "error_guards.h"

#include "gpu_network.h"
#include "p2p_uni_profiler.h"

int main(int argc, char* argv[]) {
    GPUNetwork g;
    P2PUniProfiler p;
    g.Initialize();

    if (g.comm_rank_ == 0) {
        p.Initialize(&g);

        p.ProfileOperation();
        p.GatherResults();
        p.PrintResults();

        p.Cleanup();
    }

    g.Cleanup();
    MPI_Finalize();
    return 0;
}