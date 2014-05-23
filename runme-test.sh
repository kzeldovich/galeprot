nvcc -arch=sm_30 -c kernels.cu
echo "kernels are compiled"
nvcc -arch=sm_30 -lcublas -c galelib.cu
echo "cuda wrappers are compiled"
g++ -c main.cpp
echo "example cpp code is compiled"
g++ main.o galelib.o kernels.o -o build/run -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas
echo "everything is linked and ready to go ..."
rm *.o
echo "object files are deleted to make dir look cleaner ..."

# ./build/run MJ96_letters.d c-10k-64.d 10000 faces-10k-64.d 10000 4 sequences64.in 8192 bind_pairs.in 10000 fold.out bind.out
