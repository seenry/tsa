#include "mpi.h"
#include "nccl.h"

#include "error_guards.h"

#include "gpu_network.h"
#include "p2p_uni_profiler.h"

int main(int argc, char* argv[]) {
    GPUNetwork g;
    P2PUniProfiler p;
    g.Initialize();
    p.Initialize(&g);

    p.ProfileOperation();

    p.Cleanup();
    g.Cleanup();
    return 0;
}