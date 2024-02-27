CC = nvcc

NCCL ?= /projects/p31538/nccl/build
MPI ?= /hpc/software/mpi/openmpi-4.1.4-gcc-11.2.0

INC = -I./include -I$(NCCL)/include -I$(MPI)/include
LNK = -L$(MPI)/lib -lmpi -L$(NCCL)/lib -lnccl -lm

links: build/main.o build/error_guards.o build/gpu_network.o build/p2p_uni_profiler.o build/link_cov_profiler.o
	$(CC) $(LNK) $^ -o $@

build/%.o: src/%.cpp | subdirs
	$(CC) $(INC) -c $^ -o $@

.PHONY: clean subdirs

clean:
	rm -f links build/*.o p2p_uni.csv

subdirs:
	mkdir -p build

