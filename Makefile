NVCC=nvcc
NVCCARCH=-arch=sm_30
NVCCFLAGS=$(NVCCARCH) -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcudadevrt
CC=g++
AR=ar

SRCDIR=src/

galeprotlib: $(SRCDIR)kernels.cu $(SRCDIR)galelib.cu
	mkdir -p lib
	rm -f *.o *.a
	$(NVCC) $(NVCCARCH)  -c $(SRCDIR)kernels.cu 
	$(NVCC) $(NVCCARCH)  -c $(SRCDIR)galelib.cu
	$(NVCC) $(NVCCARCH) -lib kernels.o galelib.o -o libgaleprot.a
	mv libgaleprot.a lib
	rm -f *.o *.a
#	rm kernels.o galelib.o
#	g++ -Iinclude -c main.cpp
#	g++ -o runme main.o -Llib -lgaleprot -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcudadevrt
#	g++ -o runme main.o kernels.o galelib.o -L. -lgaleprot -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcudadevrt
	
install:
	mkdir include
	cp $(SRCDIR)/galelib.h include
				