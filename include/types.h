#pragma once

#include "mpi.h"

typedef struct {
    int* sizes;
    int* ranks;
    int nrings;
    int* ring_sizes;
    int** rings;
    MPI_Comm* mpi_comms;
    ncclComm_t* nccl_comms;
    char has_ring;
} tsa_info;

#define N_MPI_COMMS 4
#define MPI_GLOB_COMM 4 // same # as N_MPI_COMMS

#define MPI_NTRA_COMM 0
#define MPI_NTER_COMM 1
#define MPI_RING_COMM 2
#define NOT_USELESS 3

#define NCCL_G_COMM 0
#define NCCL_R_COMM 1
