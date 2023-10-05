CC = nvcc

NCCL ?= /home/ubuntu/nccl/build

INC = -I./include -I$(NCCL)/include -I/usr/lib/x86_64-linux-gnu/openmpi/include
LNK = -L$(NCCL)/lib -lmpi -lnccl -lm

bin: build/main.o build/gpu_network.o
	$(CC) $(LNK) $^ -o $@

build/%.o: src/%.cpp | subdirs
	$(CC) $(INC) -c $^ -o $@

.PHONY: clean subdirs

clean:
	rm -f bin build/*.o

subdirs:
	mkdir -p build
