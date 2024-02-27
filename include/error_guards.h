#pragma once
#include <stdlib.h>
#include <stdio.h>

#define CUDACHECK(cmd) do { \
    cudaError_t err = cmd; \
    if (err != cudaSuccess) { \
        printf("Failed: CUDA error %s:%d '%s'\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define NCCLCHECK(cmd) do { \
    ncclResult_t res = cmd; \
    if (res != ncclSuccess) { \
        printf("Failed: NCCL error %s:%d '%s'\n", \
            __FILE__, __LINE__, ncclGetErrorString(res)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define MPICHECK(cmd) do { \
    int e = cmd; \
    if (e != MPI_SUCCESS) { \
        printf("Failed: MPI error %s:%d '%d'\n", \
            __FILE__, __LINE__, e); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

void CheckAlloc(void* ptr, const char* blame);
