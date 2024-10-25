#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>

#include "nccl.h"
#include "mpi.h"

#include "types.h"
#include "err_guard.h"

void InitMPIComms(tsa_info* ti_);
void InitNCCLComms(tsa_info* ti_);
void UninitNCCLComms(tsa_info* ti_);
void ParseInput(tsa_info* ti_, int argc_, const char* argv_[]);
void DestroyTI(tsa_info* ti_);

int main(int argc, const char* argv[]) {    
    tsa_info* ti = (tsa_info*) malloc(sizeof(tsa_info));
    ALLOCCHECK(ti);
    
    ParseInput(ti, argc, argv);

    MPICHECK(MPI_Init(NULL, NULL));
    InitMPIComms(ti);
    if (ti->has_ring) {
        InitNCCLComms(ti);
        UninitNCCLComms(ti);
    }

    // if (ti->ranks[MPI_GLOB_COMM] == 0) {
    //     for (int i = 0; i < ti->nrings; i++) {
    //         printf("Ring %d:\n  ", i);
    //         for (int j = 0; j < ti->ring_sizes[i]; j++) {
    //             printf("%d ", ti->rings[i][j]);
    //         }
    //         printf("\n\n");
    //     }
    // }

    DestroyTI(ti);
    MPICHECK(MPI_Finalize());    
    return 0;
}

void ParseInput(tsa_info* ti_, int argc_, const char* argv_[]) {
    ti_->nrings = 0;
    for (int i = 1; i < argc_; i++) {
        ti_->nrings++;
        long int ring_size = strtol(argv_[i], NULL, 10);
        i += ring_size;
    }

    ti_->ring_sizes = (int*) malloc(sizeof(int) * ti_->nrings);
    ti_->rings = (int**) malloc(sizeof(int*) * ti_->nrings);
    ALLOCCHECK(ti_->ring_sizes);
    ALLOCCHECK(ti_->rings);

    int ridx = 0;
    for (int i = 1; i < argc_; i++) {
        long int ring_size = strtol(argv_[i], NULL, 10);
        ti_->ring_sizes[ridx] = (int) ring_size;
        ti_->rings[ridx] = (int*) malloc(sizeof(int) * ti_->ring_sizes[ridx]);
        ALLOCCHECK(ti_->rings[ridx]);
        for (int j = 0; j < ti_->ring_sizes[ridx]; j++) {
            long int member = strtol(argv_[++i], NULL, 10);
            ti_->rings[ridx][j] = (int) member;
        }
        ridx++;
    }
}

void InitMPIComms(tsa_info* ti_) {
    ti_->sizes = (int*) malloc(sizeof(int) * (N_MPI_COMMS + 1));
    ti_->ranks = (int*) malloc(sizeof(int) * (N_MPI_COMMS + 1));
    ALLOCCHECK(ti_->sizes);
    ALLOCCHECK(ti_->ranks);

    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, ti_->sizes + MPI_GLOB_COMM));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, ti_->ranks + MPI_GLOB_COMM));

    int* hosthashes = (int*) malloc(sizeof(int) * ti_->sizes[MPI_GLOB_COMM]);
    char* hostname = (char*) malloc(sizeof(char) * 1024);
    ALLOCCHECK(hosthashes);
    ALLOCCHECK(hostname);

    gethostname(hostname, 1024);
    for (int i = 0; i < 1024; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            break;
        }
    }
    uint64_t hh_ = 5381;
    for (int i = 0 ; hostname[i] != '\0'; i++) {
        hh_ = ((hh_ << 5) + hh_) ^ hostname[i];
    }
    int hh = 0x7fffffff & (hh_ ^ (hh_ >> 32));
    hosthashes[ti_->ranks[MPI_GLOB_COMM]] = hh;
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hosthashes, sizeof(int), MPI_BYTE, MPI_COMM_WORLD));

    int local_rank = 0;
    for (int i = 0; i < ti_->sizes[MPI_GLOB_COMM]; i++) {
        if (i == ti_->ranks[MPI_GLOB_COMM]) break;
        if (hosthashes[i] == hosthashes[ti_->ranks[MPI_GLOB_COMM]]) local_rank++;
    }

    ti_->mpi_comms = (MPI_Comm*) malloc(N_MPI_COMMS * sizeof(MPI_Comm));
    ALLOCCHECK(ti_->mpi_comms);

    MPICHECK(MPI_Comm_split(
        MPI_COMM_WORLD,
        hh, local_rank,
        &ti_->mpi_comms[MPI_NTRA_COMM]
    ));

    MPICHECK(MPI_Comm_split(
        MPI_COMM_WORLD,
        local_rank == 0, ti_->ranks[MPI_GLOB_COMM],
        &ti_->mpi_comms[MPI_NTER_COMM]
    ));
    
    if (local_rank == 0) {
        MPICHECK(MPI_Comm_rank(ti_->mpi_comms[MPI_NTER_COMM], &ti_->ranks[MPI_NTER_COMM]));
    }
    MPICHECK(MPI_Bcast(&ti_->ranks[MPI_NTER_COMM], sizeof(ti_->ranks[MPI_NTER_COMM]), MPI_BYTE, 0, ti_->mpi_comms[MPI_NTRA_COMM]));

    int ngpu;
    CUDACHECK(cudaGetDeviceCount(&ngpu));
    int lb = ngpu * ti_->ranks[MPI_NTER_COMM]; // assume symmetric nodes

    int my_ring = -1;
    int next_assignee = 0;
    char done = 0;
    for (int i = 0; i < ti_->nrings; i++) {
        for (int j = 0; j < ti_->ring_sizes[i]; j++) {
            if (lb <= ti_->rings[i][j] && ti_->rings[i][j] < lb + ngpu) {
                if (local_rank == next_assignee) {
                    done = 1;
                    my_ring = i;
                    CUDACHECK(cudaSetDevice(ti_->rings[i][j]));
                    break;
                }
                next_assignee++;
            }
        }
        if (done) break;
    }
    ti_->has_ring = (my_ring != -1);

    MPICHECK(MPI_Comm_split(
        MPI_COMM_WORLD,
        my_ring & 0x7fffffff, ti_->ranks[MPI_GLOB_COMM],
        &ti_->mpi_comms[MPI_RING_COMM]
    ));
    MPICHECK(MPI_Comm_split(
        MPI_COMM_WORLD,
        ti_->has_ring, ti_->ranks[MPI_GLOB_COMM],
        &ti_->mpi_comms[NOT_USELESS]
    ));

    MPICHECK(MPI_Comm_size(ti_->mpi_comms[NOT_USELESS], &ti_->sizes[NOT_USELESS]));
    MPICHECK(MPI_Comm_rank(ti_->mpi_comms[NOT_USELESS], &ti_->ranks[NOT_USELESS]));
    MPICHECK(MPI_Comm_size(ti_->mpi_comms[MPI_RING_COMM], &ti_->sizes[MPI_RING_COMM]));
    MPICHECK(MPI_Comm_rank(ti_->mpi_comms[MPI_RING_COMM], &ti_->ranks[MPI_RING_COMM]));

    free(hosthashes);
    free(hostname);
}

void InitNCCLComms(tsa_info* ti_) {
    ncclUniqueId gid, rid;

    if (ti_->ranks[NOT_USELESS] == 0) {
        NCCLCHECK(ncclGetUniqueId(&gid));
    }
    MPICHECK(MPI_Bcast(&gid, sizeof(gid), MPI_BYTE, 0, ti_->mpi_comms[NOT_USELESS]));

    if (ti_->ranks[MPI_RING_COMM] == 0) {
        NCCLCHECK(ncclGetUniqueId(&rid));
    }
    MPICHECK(MPI_Bcast(&rid, sizeof(rid), MPI_BYTE, 0, ti_->mpi_comms[MPI_RING_COMM]));

    ti_->nccl_comms = (ncclComm_t*) malloc(sizeof(ncclComm_t) * 2);
    ALLOCCHECK(ti_->nccl_comms);

    NCCLCHECK(ncclCommInitRank(ti_->nccl_comms + NCCL_G_COMM, ti_->sizes[NOT_USELESS], gid, ti_->ranks[NOT_USELESS]));
    NCCLCHECK(ncclCommInitRank(ti_->nccl_comms + NCCL_R_COMM, ti_->sizes[MPI_RING_COMM], rid, ti_->ranks[MPI_RING_COMM]));
}

void UninitNCCLComms(tsa_info* ti_) {
    NCCLCHECK(ncclCommDestroy(ti_->nccl_comms[NCCL_G_COMM]));
    NCCLCHECK(ncclCommDestroy(ti_->nccl_comms[NCCL_R_COMM]));
    free(ti_->nccl_comms);
}

void DestroyTI(tsa_info* ti_) {

    free(ti_->sizes);
    free(ti_->ranks);
    for (int i = 0; i < N_MPI_COMMS; i++) {
        MPICHECK(MPI_Comm_free(&ti_->mpi_comms[i]));
    }
    free(ti_->mpi_comms);

    for (int i = 0; i < ti_->nrings; i++) {
        free(ti_->rings[i]);
    }
    free(ti_->ring_sizes);
    free(ti_->rings);
    free(ti_);
}
