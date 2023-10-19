CC = nvcc

NCCL ?= /home/ubuntu/nccl/build

INC = -I./include -I$(NCCL)/include -I/usr/my_mpi_install/include
LNK = -L/usr/lib/x86_64-linux-gnu -lmpi++ -lmpi -L$(NCCL)/lib -lnccl -lm

bin: build/main.o build/gpu_network.o
	$(CC) $(LNK) $^ -o $@

build/%.o: src/%.cpp | subdirs
	$(CC) $(INC) -c $^ -o $@

.PHONY: clean subdirs

clean:
	rm -f bin build/*.o

subdirs:
	mkdir -p build
