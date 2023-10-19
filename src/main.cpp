#include "mpi.h"
#include "nccl.h"

#include "gpu_network.h"
#include "error_guards.h"

int main(int argc, char* argv[]) {
    GPUNetwork g;
    g.Initialize();
    g.Point2Point();
    g.AllGather();
    g.GatherData();
    g.Print();
    
    g.Cleanup();
    return 0;
}