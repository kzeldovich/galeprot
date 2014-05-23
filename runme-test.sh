cd src
nvcc -arch=sm_30 -c kernels.cu
nvcc -arch=sm_30 -lcublas -c galelib.cu

cd ../examples
g++ -I../src -c main.cpp
g++ main.o ../src/galelib.o ../src/kernels.o -o galeprot-test -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas

./galeprot-test ../data/MJ96_letters.d ../data/c-10k-64.d 10000 ../data/faces-10k-64.d 10000 4 ../data/seq64_many.in 8192 ../data/bind_pairs.in 10000 fold.out bind.out
