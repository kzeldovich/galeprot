NVCC=nvcc
NVCCARCH=-arch=sm_20

SRCDIR=src/

galeprotlib: $(SRCDIR)kernels.cu $(SRCDIR)galelib.cu
	rm -f *.o *.a
	$(NVCC) $(NVCCARCH)  -c $(SRCDIR)kernels.cu 
	$(NVCC) $(NVCCARCH)  -c $(SRCDIR)galelib.cu
	$(NVCC) $(NVCCARCH) -lib kernels.o galelib.o -o libgaleprot.a
#	rm kernels.o galelib.o
#	g++ -Iinclude -c main.cpp
#	g++ -o runme main.o -Llib -lgaleprot -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcudadevrt
#	g++ -o runme main.o kernels.o galelib.o -L. -lgaleprot -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcudadevrt
	rm *.o
	
install:
	mkdir -p include
	cp $(SRCDIR)/galelib.h include
	mkdir -p lib
	mv libgaleprot.a lib				
