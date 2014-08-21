CC=g++
GALEPROTLIBDIR=../lib
GALEPROTINCLUDE=../include

LIBDIRS=-L$(GALEPROTLIBDIR) -L/usr/local/cuda/lib64

LIBS=-lgaleprot -lcuda -lcudart -lcublas -lcudadevrt
CFLAGS=-I$(GALEPROTINCLUDE) $(LIBS) $(LIBDIRS)

all: folding.cpp protein-design.cpp
	$(CC) folding.cpp $(CFLAGS) -o folding
	$(CC) protein-design.cpp $(CFLAGS) -o protein-design


