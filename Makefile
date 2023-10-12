CC = nvcc

NCCL ?= /home/ubuntu/nccl/build
MPI ?= /usr/my_mpi_install/lib

INC = -I./include -I$(NCCL)/include -I/usr/my_mpi_install/include
LNK = -L$(NCCL)/lib -lnccl -lm

bin: build/main.o build/gpu_network.o
	$(CC) $(LNK) $^ -o $@  -L/usr/my_mpi_install -lmpi++

build/%.o: src/%.cpp | subdirs
	$(CC) $(INC) -c $^ -o $@

.PHONY: clean subdirs

clean:
	rm -f bin build/*.o

subdirs:
	mkdir -p build
