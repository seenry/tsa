CC = nvcc

INC = -I./include -I$(NCCL_INC) -I/usr/lib/x86_64-linux-gnu/openmpi/include
LNK = -L$(NCCL_LIB) -lmpi -lnccl -lm

bin: build/main.o build/gpu_network.o
	$(CC) $(LNK) $^ -l $@

build/%.o: src/%.cpp | subdirs
	$(CC) $(INC) -c $^ -o $@

.PHONY: clean subdirs

clean:
	rm -f bin build/*.o

subdirs:
	mkdir -p build
	