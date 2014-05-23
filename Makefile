#cd src
#nvcc -arch=sm_30 -c kernels.cu
#nvcc -arch=sm_30 -lcublas -c galelib.cu
#cd ../examples
#g++ -I../src -c main.cpp
#g++ main.o ../src/galelib.o ../src/kernels.o -o galeprot-test -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas
#./galeprot-test ../data/MJ96_letters.d ../data/c-10k-64.d 10000 ../data/faces-10k-64.d 10000 4 ../data/seq64_many.in 8192 ../data/bind_pairs.in 10000 fold.out bind.out


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
	g++ -Iinclude -c main.cpp
	g++ -o runme main.o -Llib -lgaleprot -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcudadevrt
#	g++ -o runme main.o kernels.o galelib.o -L. -lgaleprot -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcudadevrt
	


install:
	mkdir include
	cp $(SRCDIR)/galelib.h include
				