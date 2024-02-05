#include "mpi.h"
#include "nccl.h"

#include "error_guards.h"

#include "gpu_network.h"
#include "textor.h"

int main(int argc, char* argv[]) {
    GPUNetwork network;
    TExtor extractor;
    network.Initialize();
    extractor.Initialize(&network);

    extractor.Extract(64);

    extractor.Destroy();
    network.Destroy();
    MPI_Finalize();
    return 0;
}