CC=g++
GALEPROTLIBDIR=../lib
GALEPROTINCLUDE=../include

LIBDIRS=-L$(GALEPROTLIBDIR) -L/usr/local/cuda/lib64

LIBS=-lgaleprot -lcuda -lcudart -lcublas -lcudadevrt
CFLAGS=-I$(GALEPROTINCLUDE) $(LIBS) $(LIBDIRS)

all: main.cpp
	$(CC) main.cpp $(CFLAGS) -o runme