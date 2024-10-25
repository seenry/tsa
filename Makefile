CC = nvcc

EXEC = tsa

NCCL_HOME ?= $(HOME)/nccl/build
MPI_HOME ?= $(HOME)/mpi

INC = -I./include -I$(NCCL_HOME)/include -I$(MPI_HOME)/include
LNK = -L$(MPI_HOME)/lib -lmpi -L$(NCCL_HOME)/lib -lnccl -lm

$(EXEC): build/main.o
	$(CC) $(LNK) $^ -o $@

build/%.o: src/%.c | subdirs
	$(CC) $(INC) -c $^ -o $@

.PHONY: clean subdirs

clean:
	rm -f $(EXEC) build/*.o

subdirs:
	mkdir -p build
