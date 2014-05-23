      
/*  GPU Accelerated Latice Evolution  */

#include<float.h>
#include "kernels.h"
#include "galelib.h"




__global__ void kernel_convert_seq_lut_v2( unsigned char *seqarray, float *seq_vector_E,
                                            float *ForceField, unsigned char *idx_lut,
                                            int MaxSeqLen, int MeanignConts, int n_seq) {
    //
    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x; int by = blockIdx.y;
    int bDx = blockDim.x; int bDy = blockDim.y;
    int Idx = bx*bDx+tx;
    int Idy = by*bDy+ty;
    //
    if (Idy < n_seq) {
        //
        unsigned char *seq = seqarray + Idy*MaxSeqLen;
        //
        if (Idx < MeanignConts) {
            //
            uchar2 idx_lut_pair = reinterpret_cast<uchar2*>(idx_lut)[Idx];
            unsigned char monomer1 = seq[ idx_lut_pair.x ];
            unsigned char monomer2 = seq[ idx_lut_pair.y ];
            //
            seq_vector_E[Idy*MeanignConts + Idx] = ForceField[ ALPHABET*monomer1 + monomer2 ];
        }
    }
}



/////////////// my kernel here - to replace following one and etc. //////////////////////////
__global__ void sub_exp_reduction(   float * mat,
                                    const float * vec,
                                    float * out,
                                    const int col_str,
                                    const int row_seq,
                                    const float coeff){
    //
    // thread and block coordinates ...
    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x; int by = blockIdx.y;
    int bDx = blockDim.x; int bDy = blockDim.y;
    int gDx = gridDim.x; //int gDy = gridDim.y;
    //
    int Idx = bDx * bx + tx;
    int Idy = bDy * by + ty;
    int gridSize = gDx * bDx;
    //
    float subtract;
    if (Idy < row_seq) {
        subtract = vec[Idy];
    }
    //
    float pre_sum = 0.0f;
    //
    float2 vec2_load;
    float  vec1_load;
    //
    // massive coalsced loading ...
    for (int i = 0; i < WORK_str; i++) {
        if (Idy < row_seq) {
            //
            if (2*Idx+1 < col_str) {
                vec2_load = reinterpret_cast<float2*>(mat)[Idy*(col_str>>1) + Idx];
                pre_sum = pre_sum    + __expf(coeff*(vec2_load.x - subtract))
                                     + __expf(coeff*(vec2_load.y - subtract));
            } else if (2*Idx < col_str) {
                vec1_load = mat[Idy*col_str + 2*Idx];
                pre_sum = pre_sum + __expf(coeff*(vec1_load - subtract));
            }
        }
        //
        Idx += gridSize;
    }
    //
    __syncthreads();
    //
    // using register shuffling within the warp - blazing fast!
    pre_sum += __shfl_down(pre_sum,8,16);
    pre_sum += __shfl_down(pre_sum,4,16);
    pre_sum += __shfl_down(pre_sum,2,16);
    pre_sum += __shfl_down(pre_sum,1,16);
    // store result back to global ...
    // reduction of the whole block resides in the 0th element ...
    if ((tx == 0) && (Idy < row_seq)) { atomicAdd(&out[Idy], pre_sum); }
    //
}
/////////////// my kernel here //////////////////////////



// __global__ void old_min_reduction2D(float * mat, float * out, int col_str, int row_seq){
//     //
//     // thread and block coordinates ...
//     int tx = threadIdx.x; int ty = threadIdx.y;
//     int bx = blockIdx.x; int by = blockIdx.y;
//     int bDx = blockDim.x; int bDy = blockDim.y;
//     int gDx = gridDim.x; //int gDy = gridDim.y;
//     //
//     int Idx = bDx * bx + tx;
//     int Idy = bDy * by + ty;
//     int gridSize = gDx * bDx;
//     //
//     float2 vec2_load;
//     float  vec1_load;
//     float pre_accum = FLT_MAX;
//     // float pre_accum;
//     //
//     // massive coalsced loading ...
//     for (int i = 0; i < WORK_str; i++) {
//         //
//         if (Idy < row_seq) {
//             if (2*Idx+1 < col_str) {
//                 vec2_load = reinterpret_cast<float2*>(mat)[Idy*(col_str>>1) + Idx];
//                 pre_accum = min(pre_accum, vec2_load.x);
//                 pre_accum = min(pre_accum, vec2_load.y);
//             } else if (2*Idx < col_str) {
//                 vec1_load = mat[Idy*col_str + 2*Idx];
//                 pre_accum = min(pre_accum, vec1_load);
//             }
//         }
//         //
//         Idx += gridSize;
//     }
//     //
//     __syncthreads();
//     //
//     // using register shuffling within the warp - blazing fast!
//     pre_accum = min(pre_accum, __shfl_down(pre_accum,8,16));
//     pre_accum = min(pre_accum, __shfl_down(pre_accum,4,16));
//     pre_accum = min(pre_accum, __shfl_down(pre_accum,2,16));
//     pre_accum = min(pre_accum, __shfl_down(pre_accum,1,16));
//     // store result back to global ...
//     //
//     // for all-negative numbers use this ...
//     if ((tx == 0) && (Idy < row_seq)) { atomicMax((unsigned int *)&out[Idy], (unsigned int)__float_as_int(pre_accum)); } // no atomicMin for float so far ...
//     // // for all-positive numbers use this ...
//     // if (tx == 0) { atomicMin((int *)&out[Idy], __float_as_int(pre_accum)); } // no atomicMin for float so far ...
//     //
// }



__device__ void lock(int *pmutex) {
    while(atomicCAS(pmutex, 0, 1) != 0);
}

__device__ void unlock(int *pmutex) {
    atomicExch(pmutex, 0);
}



__global__ void experimental_reduction2D(int *mutexes, float * mat, float * out, int * out_idx, int col_str, int row_seq){
    //
    // thread and block coordinates ...
    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x; int by = blockIdx.y;
    int bDx = blockDim.x; int bDy = blockDim.y;
    int gDx = gridDim.x; //int gDy = gridDim.y;
    //
    int Idx = bDx * bx + tx;
    int Idy = bDy * by + ty;
    int gridSize = gDx * bDx;
    // int index_to_load;
    //
    float2 vec2_load;
    float  vec1_load;
    float pre_accum = FLT_MAX;
    float pre_accum_tmp;
    int min_index=0;
    int min_index_tmp;
    // float pre_accum;
    //
    // massive coalsced loading ...
    for (int i = 0; i < WORK_str; i++) {
        //
        if (Idy < row_seq) {
            if (2*Idx+1 < col_str) {
                // index_to_load = Idy*(col_str>>1) + Idx;
                vec2_load = reinterpret_cast<float2*>(mat)[Idy*(col_str>>1) + Idx];
                if (pre_accum > vec2_load.x){pre_accum = vec2_load.x; min_index = 2*Idx+0;};
                if (pre_accum > vec2_load.y){pre_accum = vec2_load.y; min_index = 2*Idx+1;};
                // pre_accum = min(pre_accum, vec2_load.y);
            } else if (2*Idx < col_str) {
                vec1_load = mat[Idy*col_str + 2*Idx];
                if (pre_accum > vec1_load){pre_accum = vec1_load; min_index = 2*Idx;};
                // pre_accum = min(pre_accum, vec1_load);
            }
        }
        Idx += gridSize;
    }
    __syncthreads();
    // using register shuffling within the warp - blazing fast!
    pre_accum_tmp=__shfl_down(pre_accum,8,16);
    min_index_tmp=__shfl_down(min_index,8,16);
    if (pre_accum > pre_accum_tmp){ pre_accum=pre_accum_tmp; min_index=min_index_tmp;};
    pre_accum_tmp=__shfl_down(pre_accum,4,16);
    min_index_tmp=__shfl_down(min_index,4,16);
    if (pre_accum > pre_accum_tmp){ pre_accum=pre_accum_tmp; min_index=min_index_tmp;};
    pre_accum_tmp=__shfl_down(pre_accum,2,16);
    min_index_tmp=__shfl_down(min_index,2,16);
    if (pre_accum > pre_accum_tmp){ pre_accum=pre_accum_tmp; min_index=min_index_tmp;};
    pre_accum_tmp=__shfl_down(pre_accum,1,16);
    min_index_tmp=__shfl_down(min_index,1,16);
    if (pre_accum > pre_accum_tmp){ pre_accum=pre_accum_tmp; min_index=min_index_tmp;};
    //
    //
    if ((tx == 0) && (Idy < row_seq)) {
        lock(&mutexes[Idy]);
        float old_val = out[Idy];
        if (pre_accum < old_val) {
            out[Idy] = pre_accum;
            out_idx[Idy] = min_index;
        }
        unlock(&mutexes[Idy]);
    }
}



__global__ void min_reduction2D_unified(float * mat, float * out_tmp, int * str_idx_tmp, int col_str, int row_seq, int col_tmp){
    //
    // thread and block coordinates ...
    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x; int by = blockIdx.y;
    int bDx = blockDim.x; int bDy = blockDim.y;
    int gDx = gridDim.x; //int gDy = gridDim.y;
    //
    int Idx = bDx * bx + tx;
    int Idy = bDy * by + ty;
    int gridSize = gDx * bDx;
    // int index_to_load;
    //
    float2 vec2_load;
    float  vec1_load;
    float pre_accum = FLT_MAX;
    float pre_accum_tmp;
    int min_index=0;
    int min_index_tmp;
    // float pre_accum;
    //
    // massive coalsced loading ...
    for (int i = 0; i < WORK_str; i++) {
        //
        if (Idy < row_seq) {
            if (2*Idx+1 < col_str) {
                // index_to_load = Idy*(col_str>>1) + Idx;
                vec2_load = reinterpret_cast<float2*>(mat)[Idy*(col_str>>1) + Idx];
                if (pre_accum > vec2_load.x){pre_accum = vec2_load.x; min_index = 2*Idx+0;};
                if (pre_accum > vec2_load.y){pre_accum = vec2_load.y; min_index = 2*Idx+1;};
                // pre_accum = min(pre_accum, vec2_load.y);
            } else if (2*Idx < col_str) {
                vec1_load = mat[Idy*col_str + 2*Idx];
                if (pre_accum > vec1_load){pre_accum = vec1_load; min_index = 2*Idx;};
                // pre_accum = min(pre_accum, vec1_load);
            }
        }
        Idx += gridSize;
    }
    __syncthreads();
    // using register shuffling within the warp - blazing fast!
    pre_accum_tmp=__shfl_down(pre_accum,8,16);
    min_index_tmp=__shfl_down(min_index,8,16);
    if (pre_accum > pre_accum_tmp){ pre_accum=pre_accum_tmp; min_index=min_index_tmp;};
    pre_accum_tmp=__shfl_down(pre_accum,4,16);
    min_index_tmp=__shfl_down(min_index,4,16);
    if (pre_accum > pre_accum_tmp){ pre_accum=pre_accum_tmp; min_index=min_index_tmp;};
    pre_accum_tmp=__shfl_down(pre_accum,2,16);
    min_index_tmp=__shfl_down(min_index,2,16);
    if (pre_accum > pre_accum_tmp){ pre_accum=pre_accum_tmp; min_index=min_index_tmp;};
    pre_accum_tmp=__shfl_down(pre_accum,1,16);
    min_index_tmp=__shfl_down(min_index,1,16);
    if (pre_accum > pre_accum_tmp){ pre_accum=pre_accum_tmp; min_index=min_index_tmp;};
    //
    if ((tx == 0) && (Idy < row_seq)) {
        out_tmp[Idy*col_tmp + bx] = pre_accum;
        str_idx_tmp[Idy*col_tmp + bx] = min_index;
    }
}


__global__ void min_reduction2D_oneblockx(float * in_tmp, float * out, int * str_idx_tmp, int * str_idx, int row_seq, int col_tmp){
    //
    // thread and block coordinates ...
    int tx = threadIdx.x; int ty = threadIdx.y;
    int by = blockIdx.y;
    int bDy = blockDim.y;
    int Idy = bDy * by + ty;
    //
    // float  vec1_load;
    float pre_accum = FLT_MAX;
    float pre_accum_tmp;
    int min_index = 0;
    int tx_tmp;
    //
    if (Idy < row_seq) {
        if (tx < col_tmp) {
            pre_accum = in_tmp[Idy*col_tmp + tx];
            min_index = str_idx_tmp[Idy*col_tmp + tx];
        }
    }
    //
    __syncthreads();
    //
    // using register shuffling within the warp - blazing fast!
    pre_accum_tmp = __shfl_down(pre_accum,8,16);
    tx_tmp =        __shfl_down(min_index,8,16);
    if (pre_accum > pre_accum_tmp){pre_accum = pre_accum_tmp; min_index = tx_tmp;};
    pre_accum_tmp = __shfl_down(pre_accum,4,16);
    tx_tmp =        __shfl_down(min_index,4,16);
    if (pre_accum > pre_accum_tmp){pre_accum = pre_accum_tmp; min_index = tx_tmp;};
    pre_accum_tmp = __shfl_down(pre_accum,2,16);
    tx_tmp =        __shfl_down(min_index,2,16);
    if (pre_accum > pre_accum_tmp){pre_accum = pre_accum_tmp; min_index = tx_tmp;};
    pre_accum_tmp = __shfl_down(pre_accum,1,16);
    tx_tmp =        __shfl_down(min_index,1,16);
    if (pre_accum > pre_accum_tmp){pre_accum = pre_accum_tmp; min_index = tx_tmp;};
    // store result back to global ...
    //
    if ((tx == 0) && (Idy < row_seq)) { out[Idy] = pre_accum; str_idx[Idy] = min_index; }
    //
}



__global__ void kernel_invert_Z(float *Z, int len){
    //
    int bDx = blockDim.x;
    int Idx = blockIdx.x * 2*bDx + threadIdx.x  ;
    int gridSize = gridDim.x * 2*bDx;
    //
    // massive coalsced loading ...
    for (int i = 0; i < WORK_str; i++) {
        if (Idx + bDx < len) {
            Z[Idx]          = 1.0f/Z[Idx];
            Z[Idx + bDx]    = 1.0f/Z[Idx + bDx];
        } else if (Idx < len) {
            Z[Idx]          = 1.0f/Z[Idx];
        }
        Idx += gridSize;
    }
}

// __global__ void kernel_invert_Z_plus(float *Z, int len){
//     //
//     int tx = threadIdx.x; int ty = threadIdx.y;
//     int bx = blockIdx.x; int by = blockIdx.y;
//     int bDx = blockDim.x; int bDy = blockDim.y;
//     int gDx = gridDim.x; // int gDy = gridDim.y;
//     int Idx = bx*bDx+tx; int Idy = by*bDy+ty;
//     //
//     int Wdx = gDx*bDx;
//     //
//     int index = min(Idx + Idy*Wdx,len);
//     //
//     Z[index] = 1.0f/Z[index];
//     //
// }

///////////////////////
// for binding purposes only ...
// __global__ void sort_and_copy_sequences(unsigned char *out, unsigned char *in, int *order, int len, int seq_len){
//     //
//     int tx = threadIdx.x; int ty = threadIdx.y;
//     int bx = blockIdx.x; int by = blockIdx.y;
//     int bDx = blockDim.x; int bDy = blockDim.y;
//     int Idx = bx*bDx+tx;
//     int Idy = by*bDy+ty;
//     //
//     int idy_local = min(Idy,len-1);
//     int idx_local = min(Idx,seq_len-1);
//     int origin = order[idy_local];
//     //
//     out[idy_local*seq_len+idx_local] = in[origin*seq_len+idx_local];
// }



// __global__ void grand_unified_kernel(float *ForceField, unsigned char *seqarray, unsigned char *faces_array, int *order_conf, int *order_seq, int n_pairs, int edgelen, int seq_len){
//     //
//     int tx = threadIdx.x;   int ty = threadIdx.y;   int tz = threadIdx.z;
//     int bx = blockIdx.x;    int by = blockIdx.y;    int bz = blockIdx.z;
//     int bDx = blockDim.x;   int bDy = blockDim.y;   int bDz = blockDim.z;
//     // int gDx = gridDim.x;    int gDy = gridDim.y;
//     int Idx = bx*bDx+tx;    int Idy = by*bDy+ty;    int Idz = bz*bDz+tz;
//     //
//     // kinda need edge squared ... kinda face size ...
//     int fsize = edgelen*edgelen;
//     int facearr_factor = FACES*ORIENTATIONS*fsize;
//     int factor144 = FACES*facearr_factor;
//     // kinda get two conformations to bind first ...
//     int2 pair_conf = reinterpret_cast<int2*>(order_conf)[Idz];
//     unsigned char *conf1 = &faces_array[facearr_factor*pair.x];
//     unsigned char *conf2 = &faces_array[facearr_factor*pair.y];
//     // kinda get two sequences to bind first ...
//     int2 seq_seq   = reinterpret_cast<int2*>(order_seq)[Idz];
//     unsigned char *conf1_ptr = &faces_array[facearr_factor*pair.x];
//     unsigned char *conf2_ptr = &faces_array[facearr_factor*pair.y];
// }



// __global__ void get_seq_pairwise_vector(unsigned char *seqarray, float *seqseq_vector_E,
//                                         int *order, float *ForceField,
//                                         int n_pairs, int seq_len) {
//     //
//     int tx = threadIdx.x; int ty = threadIdx.y;
//     int bx = blockIdx.x; int by = blockIdx.y;
//     int bDx = blockDim.x; int bDy = blockDim.y;
//     int Idx = bx*bDx+tx;
//     int Idy = by*bDy+ty;
//     //
//     int seqseq_vec_len = seq_len*seq_len;
//     //
//     if (Idy < n_pairs){
//         // two sequences to get their pairwise interactions ...
//         int2 a_pair = reinterpret_cast<int2*>(order)[Idy];
//         unsigned char *seq1 = seqarray + a_pair.x*seq_len;
//         unsigned char *seq2 = seqarray + a_pair.y*seq_len;
//         if (Idx < seqseq_vec_len) {
//             // Idx goes along this seqseq_vec_len interaction vector so Row And Col are: 
//             int Row = Idx/seq_len; 
//             int Col = Idx%seq_len; 
//             // now just fill in the seqseq_vector_E ... 
//             seqseq_vector_E[Idy*seqseq_vec_len + Idx] = ForceField[ ALPHABET*seq1[Row] + seq2[Col] ]; 
//         }
//     } 
// }


__global__ void get_seq_pairwise_vector(unsigned char *seqarray, float *seqseq_vector_E,
                                        int *order, float *ForceField,
                                        int n_pairs, int seq_len){
    //
    int tx = threadIdx.x; int ty = threadIdx.y;
    // int bx = blockIdx.x;
    int by = blockIdx.y;
    int bDx = blockDim.x; int bDy = blockDim.y;
    // int Idx = bx*bDx+tx;
    // int gDx = gridDim.x; // int gDy = gridDim.y;
    // squared arrangement: (bx,by) -> Idy analog would be ty + bx*bDy + by*gDx*bDy
    int Idy = by*bDy+ty;
    // int Idy = ty + bx*bDy + by*gDx*bDy;
    int tid = tx + ty*bDx;
    //
    int Idy_loc = min(Idy,n_pairs-1);
    int seqseq_vec_len = seq_len*seq_len;
    //
    //
    int2 a_pair = reinterpret_cast<int2*>(order)[Idy_loc];
    unsigned char *seq1 = seqarray + a_pair.x*seq_len;
    unsigned char *seq2 = seqarray + a_pair.y*seq_len;
    //
    //
    extern __shared__ float scratch[];
    //
    // float *FF = &scratch[0];
    // int *seq1_arr = (int *)&scratch[0]; // size is bDy*seq_len
    // int *seq2_arr = (int *)&scratch[0+bDy*seq_len]; // size is bDy*seq_len
    float *FF = &scratch[0];
    int *seq1_arr = (int *)&scratch[ALPHABET*ALPHABET]; // size is bDy*seq_len
    int *seq2_arr = (int *)&scratch[ALPHABET*ALPHABET+bDy*seq_len]; // size is bDy*seq_len
    //
    // Force fiels loading ...
    FF[tid] = ForceField[tid];
    int tid_my = min(184,tid); // up to 441 ...
    FF[256+tid_my] = ForceField[256+tid_my];
    __syncthreads();
    //
    // seq1,2_arr loading ...
    // #pragma unroll
    for (int i = 0; i < (seq_len-1)/bDx+1; i++) {
    // for (int i = 0; i < 4; i++) {
        seq1_arr[ty*seq_len+i*bDx+tx] = seq1[i*bDx+tx];
        seq2_arr[ty*seq_len+i*bDx+tx] = seq2[i*bDx+tx];
    }
    __syncthreads();
    //
    //
    if (Idy < n_pairs) {
        // #pragma unroll 16
        for (int i = 0; i < (seqseq_vec_len-1)/bDx+1; i++) {
            //
            // int index = min(i*bDx + tx,seqseq_vec_len-1); //0..15, 16..31, 32..48, ..., 40xx..4095
            int index = i*bDx + tx; //0..15, 16..31, 32..48, ..., 40xx..4095
            int row = index/seq_len;
            int col = index%seq_len;
            // seqseq_vector_E[Idy*seqseq_vec_len + index] = ForceField[ ALPHABET*seq1_arr[ty*seq_len+row] + seq2_arr[ty*seq_len+col] ];
            seqseq_vector_E[Idy*seqseq_vec_len + index] = FF[ ALPHABET*seq1_arr[ty*seq_len+row] + seq2_arr[ty*seq_len+col] ];
        }
    }
    //
}





__global__ void get_binding_spectra_coords(int *out, unsigned char *faces_array, int *order, int n_pairs, int edgelen, int seq_len){
    //
    int tx = threadIdx.x;   int ty = threadIdx.y;   int tz = threadIdx.z;
    int bx = blockIdx.x;    int by = blockIdx.y;    int bz = blockIdx.z;
    int bDx = blockDim.x;   int bDy = blockDim.y;   int bDz = blockDim.z;
    // int gDx = gridDim.x;    int gDy = gridDim.y;
    int Idx = bx*bDx+tx;    int Idy = by*bDy+ty;    int Idz = bz*bDz+tz;
    //
    // kinda need edge squared ... kinda face size ...
    int fsize = edgelen*edgelen;
    int facearr_factor = FACES*ORIENTATIONS*fsize;
    int factor144 = FACES*facearr_factor;
    // kinda get two conformations to bind first ...
    int2 pair = reinterpret_cast<int2*>(order)[Idz];
    unsigned char *conf1_ptr = &faces_array[facearr_factor*pair.x];
    unsigned char *conf2_ptr = &faces_array[facearr_factor*pair.y];
    //
    // local Idx for not to exceed the edge*edge or the face size ...
    int local_Idx = min(Idx,fsize-1);
    //
    // each face in FACES=6 and ORIENTATIONS=4 occupy 6*4*edge*edge elements of "unsigned char" ...
    extern __shared__ int conf1_buffer[];
    // all faces in upwards orientation ...
    size_t z_offset = tz*FACES*bDx;
    // //
    // // conf2_ptr[Idy*FACES*fsize + 0*fsize + local_Idx]
    // //
    int *f0up = &conf1_buffer[z_offset + 0*bDx];
    int *f1up = &conf1_buffer[z_offset + 1*bDx];
    int *f2up = &conf1_buffer[z_offset + 2*bDx];
    int *f3up = &conf1_buffer[z_offset + 3*bDx];
    int *f4up = &conf1_buffer[z_offset + 4*bDx];
    int *f5up = &conf1_buffer[z_offset + 5*bDx];
    // loading stuff up ...
    f0up[tx] = (int)conf1_ptr[0*ORIENTATIONS*fsize + local_Idx];
    f1up[tx] = (int)conf1_ptr[1*ORIENTATIONS*fsize + local_Idx];
    f2up[tx] = (int)conf1_ptr[2*ORIENTATIONS*fsize + local_Idx];
    f3up[tx] = (int)conf1_ptr[3*ORIENTATIONS*fsize + local_Idx];
    f4up[tx] = (int)conf1_ptr[4*ORIENTATIONS*fsize + local_Idx];
    f5up[tx] = (int)conf1_ptr[5*ORIENTATIONS*fsize + local_Idx];
    // waiting for all loads to complete ...
    __syncthreads();
    // // to be out is 144*edge*edge X #_pairs huge thing ...
    // // out[factor144*Idz + 0*facearr_factor + Idy*FACES*fsize + 0*fsize + local_Idx] = seq_len*f0up[tx] + conf2_ptr[Idy*FACES*fsize + 0*fsize + local_Idx];
    // // out[factor144*Idz + 0*facearr_factor + Idy*FACES*fsize + 1*fsize + local_Idx] = seq_len*f1up[tx] + conf2_ptr[Idy*FACES*fsize + 1*fsize + local_Idx];
    // // out[factor144*Idz + 0*facearr_factor + Idy*FACES*fsize + 2*fsize + local_Idx] = seq_len*f2up[tx] + conf2_ptr[Idy*FACES*fsize + 2*fsize + local_Idx];
    // // out[factor144*Idz + 0*facearr_factor + Idy*FACES*fsize + 3*fsize + local_Idx] = seq_len*f3up[tx] + conf2_ptr[Idy*FACES*fsize + 3*fsize + local_Idx];
    // // out[factor144*Idz + 0*facearr_factor + Idy*FACES*fsize + 4*fsize + local_Idx] = seq_len*f4up[tx] + conf2_ptr[Idy*FACES*fsize + 4*fsize + local_Idx];
    // // out[factor144*Idz + 0*facearr_factor + Idy*FACES*fsize + 5*fsize + local_Idx] = seq_len*f5up[tx] + conf2_ptr[Idy*FACES*fsize + 5*fsize + local_Idx];
    // //
    //
    // CYCLIC PERMUTATIONS ...
    //
    // for (int i = 0; i < FACES; i++) {
    //     for (int j = 0; j < FACES; j++) {
    //         // for (int k = 0; k < count; k++) {
    //             //
    //             out[factor144*Idz + i*facearr_factor + Idy*FACES*fsize + ((0+j)%FACES)*fsize + local_Idx] = seq_len*f0up[tx] + conf2_ptr[Idy*FACES*fsize + ((0+j)%FACES)*fsize + local_Idx];
    //             out[factor144*Idz + i*facearr_factor + Idy*FACES*fsize + ((1+j)%FACES)*fsize + local_Idx] = seq_len*f1up[tx] + conf2_ptr[Idy*FACES*fsize + ((1+j)%FACES)*fsize + local_Idx];
    //             out[factor144*Idz + i*facearr_factor + Idy*FACES*fsize + ((2+j)%FACES)*fsize + local_Idx] = seq_len*f2up[tx] + conf2_ptr[Idy*FACES*fsize + ((2+j)%FACES)*fsize + local_Idx];
    //             out[factor144*Idz + i*facearr_factor + Idy*FACES*fsize + ((3+j)%FACES)*fsize + local_Idx] = seq_len*f3up[tx] + conf2_ptr[Idy*FACES*fsize + ((3+j)%FACES)*fsize + local_Idx];
    //             out[factor144*Idz + i*facearr_factor + Idy*FACES*fsize + ((4+j)%FACES)*fsize + local_Idx] = seq_len*f4up[tx] + conf2_ptr[Idy*FACES*fsize + ((4+j)%FACES)*fsize + local_Idx];
    //             out[factor144*Idz + i*facearr_factor + Idy*FACES*fsize + ((5+j)%FACES)*fsize + local_Idx] = seq_len*f5up[tx] + conf2_ptr[Idy*FACES*fsize + ((5+j)%FACES)*fsize + local_Idx];
    //         // }
    //     }
    // }
    //
    out[factor144*Idz + 0*facearr_factor + Idy*FACES*fsize + 0*fsize + local_Idx] = seq_len*f0up[tx] + conf2_ptr[Idy*FACES*fsize + 0*fsize + local_Idx];
    out[factor144*Idz + 0*facearr_factor + Idy*FACES*fsize + 1*fsize + local_Idx] = seq_len*f1up[tx] + conf2_ptr[Idy*FACES*fsize + 1*fsize + local_Idx];
    out[factor144*Idz + 0*facearr_factor + Idy*FACES*fsize + 2*fsize + local_Idx] = seq_len*f2up[tx] + conf2_ptr[Idy*FACES*fsize + 2*fsize + local_Idx];
    out[factor144*Idz + 0*facearr_factor + Idy*FACES*fsize + 3*fsize + local_Idx] = seq_len*f3up[tx] + conf2_ptr[Idy*FACES*fsize + 3*fsize + local_Idx];
    out[factor144*Idz + 0*facearr_factor + Idy*FACES*fsize + 4*fsize + local_Idx] = seq_len*f4up[tx] + conf2_ptr[Idy*FACES*fsize + 4*fsize + local_Idx];
    out[factor144*Idz + 0*facearr_factor + Idy*FACES*fsize + 5*fsize + local_Idx] = seq_len*f5up[tx] + conf2_ptr[Idy*FACES*fsize + 5*fsize + local_Idx];
    //
    out[factor144*Idz + 1*facearr_factor + Idy*FACES*fsize + 0*fsize + local_Idx] = seq_len*f5up[tx] + conf2_ptr[Idy*FACES*fsize + 0*fsize + local_Idx];
    out[factor144*Idz + 1*facearr_factor + Idy*FACES*fsize + 1*fsize + local_Idx] = seq_len*f0up[tx] + conf2_ptr[Idy*FACES*fsize + 1*fsize + local_Idx];
    out[factor144*Idz + 1*facearr_factor + Idy*FACES*fsize + 2*fsize + local_Idx] = seq_len*f1up[tx] + conf2_ptr[Idy*FACES*fsize + 2*fsize + local_Idx];
    out[factor144*Idz + 1*facearr_factor + Idy*FACES*fsize + 3*fsize + local_Idx] = seq_len*f2up[tx] + conf2_ptr[Idy*FACES*fsize + 3*fsize + local_Idx];
    out[factor144*Idz + 1*facearr_factor + Idy*FACES*fsize + 4*fsize + local_Idx] = seq_len*f3up[tx] + conf2_ptr[Idy*FACES*fsize + 4*fsize + local_Idx];
    out[factor144*Idz + 1*facearr_factor + Idy*FACES*fsize + 5*fsize + local_Idx] = seq_len*f4up[tx] + conf2_ptr[Idy*FACES*fsize + 5*fsize + local_Idx];
    //
    out[factor144*Idz + 2*facearr_factor + Idy*FACES*fsize + 0*fsize + local_Idx] = seq_len*f4up[tx] + conf2_ptr[Idy*FACES*fsize + 0*fsize + local_Idx];
    out[factor144*Idz + 2*facearr_factor + Idy*FACES*fsize + 1*fsize + local_Idx] = seq_len*f5up[tx] + conf2_ptr[Idy*FACES*fsize + 1*fsize + local_Idx];
    out[factor144*Idz + 2*facearr_factor + Idy*FACES*fsize + 2*fsize + local_Idx] = seq_len*f0up[tx] + conf2_ptr[Idy*FACES*fsize + 2*fsize + local_Idx];
    out[factor144*Idz + 2*facearr_factor + Idy*FACES*fsize + 3*fsize + local_Idx] = seq_len*f1up[tx] + conf2_ptr[Idy*FACES*fsize + 3*fsize + local_Idx];
    out[factor144*Idz + 2*facearr_factor + Idy*FACES*fsize + 4*fsize + local_Idx] = seq_len*f2up[tx] + conf2_ptr[Idy*FACES*fsize + 4*fsize + local_Idx];
    out[factor144*Idz + 2*facearr_factor + Idy*FACES*fsize + 5*fsize + local_Idx] = seq_len*f3up[tx] + conf2_ptr[Idy*FACES*fsize + 5*fsize + local_Idx];
    //
    out[factor144*Idz + 3*facearr_factor + Idy*FACES*fsize + 0*fsize + local_Idx] = seq_len*f3up[tx] + conf2_ptr[Idy*FACES*fsize + 0*fsize + local_Idx];
    out[factor144*Idz + 3*facearr_factor + Idy*FACES*fsize + 1*fsize + local_Idx] = seq_len*f4up[tx] + conf2_ptr[Idy*FACES*fsize + 1*fsize + local_Idx];
    out[factor144*Idz + 3*facearr_factor + Idy*FACES*fsize + 2*fsize + local_Idx] = seq_len*f5up[tx] + conf2_ptr[Idy*FACES*fsize + 2*fsize + local_Idx];
    out[factor144*Idz + 3*facearr_factor + Idy*FACES*fsize + 3*fsize + local_Idx] = seq_len*f0up[tx] + conf2_ptr[Idy*FACES*fsize + 3*fsize + local_Idx];
    out[factor144*Idz + 3*facearr_factor + Idy*FACES*fsize + 4*fsize + local_Idx] = seq_len*f1up[tx] + conf2_ptr[Idy*FACES*fsize + 4*fsize + local_Idx];
    out[factor144*Idz + 3*facearr_factor + Idy*FACES*fsize + 5*fsize + local_Idx] = seq_len*f2up[tx] + conf2_ptr[Idy*FACES*fsize + 5*fsize + local_Idx];
    //
    out[factor144*Idz + 4*facearr_factor + Idy*FACES*fsize + 0*fsize + local_Idx] = seq_len*f2up[tx] + conf2_ptr[Idy*FACES*fsize + 0*fsize + local_Idx];
    out[factor144*Idz + 4*facearr_factor + Idy*FACES*fsize + 1*fsize + local_Idx] = seq_len*f3up[tx] + conf2_ptr[Idy*FACES*fsize + 1*fsize + local_Idx];
    out[factor144*Idz + 4*facearr_factor + Idy*FACES*fsize + 2*fsize + local_Idx] = seq_len*f4up[tx] + conf2_ptr[Idy*FACES*fsize + 2*fsize + local_Idx];
    out[factor144*Idz + 4*facearr_factor + Idy*FACES*fsize + 3*fsize + local_Idx] = seq_len*f5up[tx] + conf2_ptr[Idy*FACES*fsize + 3*fsize + local_Idx];
    out[factor144*Idz + 4*facearr_factor + Idy*FACES*fsize + 4*fsize + local_Idx] = seq_len*f0up[tx] + conf2_ptr[Idy*FACES*fsize + 4*fsize + local_Idx];
    out[factor144*Idz + 4*facearr_factor + Idy*FACES*fsize + 5*fsize + local_Idx] = seq_len*f1up[tx] + conf2_ptr[Idy*FACES*fsize + 5*fsize + local_Idx];
    //
    out[factor144*Idz + 5*facearr_factor + Idy*FACES*fsize + 0*fsize + local_Idx] = seq_len*f1up[tx] + conf2_ptr[Idy*FACES*fsize + 0*fsize + local_Idx];
    out[factor144*Idz + 5*facearr_factor + Idy*FACES*fsize + 1*fsize + local_Idx] = seq_len*f2up[tx] + conf2_ptr[Idy*FACES*fsize + 1*fsize + local_Idx];
    out[factor144*Idz + 5*facearr_factor + Idy*FACES*fsize + 2*fsize + local_Idx] = seq_len*f3up[tx] + conf2_ptr[Idy*FACES*fsize + 2*fsize + local_Idx];
    out[factor144*Idz + 5*facearr_factor + Idy*FACES*fsize + 3*fsize + local_Idx] = seq_len*f4up[tx] + conf2_ptr[Idy*FACES*fsize + 3*fsize + local_Idx];
    out[factor144*Idz + 5*facearr_factor + Idy*FACES*fsize + 4*fsize + local_Idx] = seq_len*f5up[tx] + conf2_ptr[Idy*FACES*fsize + 4*fsize + local_Idx];
    out[factor144*Idz + 5*facearr_factor + Idy*FACES*fsize + 5*fsize + local_Idx] = seq_len*f0up[tx] + conf2_ptr[Idy*FACES*fsize + 5*fsize + local_Idx];
    //    
}



__global__ void get_binding_spectra_coords_FF(int *out, unsigned char *faces_array, unsigned char *seqarray, int *order_conf, int *order_seq, int n_pairs, int edgelen, int seq_len){
    //
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    //
    int bx = blockIdx.x;
    // int by = blockIdx.y;
    int bz = blockIdx.z;
    //
    int bDx = blockDim.x;
    // int bDy = blockDim.y;
    int bDz = blockDim.z;
    //
    // int gDx = gridDim.x;    int gDy = gridDim.y;
    int Idx = bx*bDx+tx;
    // int Idy = by*bDy+ty;
    int Idz = bz*bDz+tz;
    //
    // kinda need edge squared ... kinda face size ...
    int fsize = edgelen*edgelen;
    int facearr_factor = FACES*ORIENTATIONS*fsize;
    int factor144 = FACES*facearr_factor;
    // kinda get two conformations to bind first ...
    int2 pair = reinterpret_cast<int2*>(order_conf)[Idz];
    unsigned char *conf1_ptr = &faces_array[facearr_factor*pair.x];
    unsigned char *conf2_ptr = &faces_array[facearr_factor*pair.y];
    //
    int2 pair_seq = reinterpret_cast<int2*>(order_seq)[Idz];
    unsigned char *seq1 = &seqarray[seq_len*pair_seq.x];
    unsigned char *seq2 = &seqarray[seq_len*pair_seq.y];
    //
    // local Idx for not to exceed the edge*edge or the face size ...
    int local_Idx = min(Idx,fsize-1);
    //
    // each face in FACES=6 and ORIENTATIONS=4 occupy 6*4*edge*edge elements of "unsigned char" ...
    extern __shared__ int conf12_and_seq12_buff[];
    // all faces in upwards orientation ...
    size_t z_offset = tz*FACES*bDx;
    //
    int *f0up = &conf12_and_seq12_buff[z_offset + 0*bDx];
    int *f1up = &conf12_and_seq12_buff[z_offset + 1*bDx];
    int *f2up = &conf12_and_seq12_buff[z_offset + 2*bDx];
    int *f3up = &conf12_and_seq12_buff[z_offset + 3*bDx];
    int *f4up = &conf12_and_seq12_buff[z_offset + 4*bDx];
    int *f5up = &conf12_and_seq12_buff[z_offset + 5*bDx];
    // loading stuff up ...
    f0up[tx] = (int)conf1_ptr[0*ORIENTATIONS*fsize + local_Idx];
    f1up[tx] = (int)conf1_ptr[1*ORIENTATIONS*fsize + local_Idx];
    f2up[tx] = (int)conf1_ptr[2*ORIENTATIONS*fsize + local_Idx];
    f3up[tx] = (int)conf1_ptr[3*ORIENTATIONS*fsize + local_Idx];
    f4up[tx] = (int)conf1_ptr[4*ORIENTATIONS*fsize + local_Idx];
    f5up[tx] = (int)conf1_ptr[5*ORIENTATIONS*fsize + local_Idx];
    //
    int conf2_offset = bDz*FACES*bDx;
    //
    // fsoX is an array of bDx*ORIENTATIONS -> face X in 4 orientations 0..3
    int *fso0 = &conf12_and_seq12_buff[conf2_offset + z_offset*ORIENTATIONS + 0*bDx*ORIENTATIONS];
    int *fso1 = &conf12_and_seq12_buff[conf2_offset + z_offset*ORIENTATIONS + 1*bDx*ORIENTATIONS];
    int *fso2 = &conf12_and_seq12_buff[conf2_offset + z_offset*ORIENTATIONS + 2*bDx*ORIENTATIONS];
    int *fso3 = &conf12_and_seq12_buff[conf2_offset + z_offset*ORIENTATIONS + 3*bDx*ORIENTATIONS];
    int *fso4 = &conf12_and_seq12_buff[conf2_offset + z_offset*ORIENTATIONS + 4*bDx*ORIENTATIONS];
    int *fso5 = &conf12_and_seq12_buff[conf2_offset + z_offset*ORIENTATIONS + 5*bDx*ORIENTATIONS];
    //
    fso0[ty*bDx + tx] = (int)conf2_ptr[0*ORIENTATIONS*fsize + ty*fsize + local_Idx];
    fso1[ty*bDx + tx] = (int)conf2_ptr[1*ORIENTATIONS*fsize + ty*fsize + local_Idx];
    fso2[ty*bDx + tx] = (int)conf2_ptr[2*ORIENTATIONS*fsize + ty*fsize + local_Idx];
    fso3[ty*bDx + tx] = (int)conf2_ptr[3*ORIENTATIONS*fsize + ty*fsize + local_Idx];
    fso4[ty*bDx + tx] = (int)conf2_ptr[4*ORIENTATIONS*fsize + ty*fsize + local_Idx];
    fso5[ty*bDx + tx] = (int)conf2_ptr[5*ORIENTATIONS*fsize + ty*fsize + local_Idx];
    //
    int seq12_offset = conf2_offset + bDz*FACES*ORIENTATIONS*bDx;
    //
    int *seq1_shared = &conf12_and_seq12_buff[seq12_offset + tz*2*seq_len + 0*seq_len];
    int *seq2_shared = &conf12_and_seq12_buff[seq12_offset + tz*2*seq_len + 1*seq_len];
    // for 64-mers our block x,y = 16,4 wotk out perfectly ...
    seq1_shared[ty*bDx + tx] = (int)seq1[ty*bDx + tx];
    seq2_shared[ty*bDx + tx] = (int)seq2[ty*bDx + tx];
    // waiting for all loads to complete ...
    __syncthreads();
    //
    // CYCLIC PERMUTATIONS ...
    out[factor144*Idz + 0*facearr_factor + 0*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f0up[tx]] + seq2_shared[fso0[ty*bDx + tx]];
    out[factor144*Idz + 0*facearr_factor + 1*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f1up[tx]] + seq2_shared[fso1[ty*bDx + tx]];
    out[factor144*Idz + 0*facearr_factor + 2*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f2up[tx]] + seq2_shared[fso2[ty*bDx + tx]];
    out[factor144*Idz + 0*facearr_factor + 3*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f3up[tx]] + seq2_shared[fso3[ty*bDx + tx]];
    out[factor144*Idz + 0*facearr_factor + 4*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f4up[tx]] + seq2_shared[fso4[ty*bDx + tx]];
    out[factor144*Idz + 0*facearr_factor + 5*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f5up[tx]] + seq2_shared[fso5[ty*bDx + tx]];
    //
    out[factor144*Idz + 1*facearr_factor + 0*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f5up[tx]] + seq2_shared[fso0[ty*bDx + tx]];
    out[factor144*Idz + 1*facearr_factor + 1*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f0up[tx]] + seq2_shared[fso1[ty*bDx + tx]];
    out[factor144*Idz + 1*facearr_factor + 2*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f1up[tx]] + seq2_shared[fso2[ty*bDx + tx]];
    out[factor144*Idz + 1*facearr_factor + 3*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f2up[tx]] + seq2_shared[fso3[ty*bDx + tx]];
    out[factor144*Idz + 1*facearr_factor + 4*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f3up[tx]] + seq2_shared[fso4[ty*bDx + tx]];
    out[factor144*Idz + 1*facearr_factor + 5*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f4up[tx]] + seq2_shared[fso5[ty*bDx + tx]];
    //
    out[factor144*Idz + 2*facearr_factor + 0*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f4up[tx]] + seq2_shared[fso0[ty*bDx + tx]];
    out[factor144*Idz + 2*facearr_factor + 1*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f5up[tx]] + seq2_shared[fso1[ty*bDx + tx]];
    out[factor144*Idz + 2*facearr_factor + 2*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f0up[tx]] + seq2_shared[fso2[ty*bDx + tx]];
    out[factor144*Idz + 2*facearr_factor + 3*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f1up[tx]] + seq2_shared[fso3[ty*bDx + tx]];
    out[factor144*Idz + 2*facearr_factor + 4*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f2up[tx]] + seq2_shared[fso4[ty*bDx + tx]];
    out[factor144*Idz + 2*facearr_factor + 5*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f3up[tx]] + seq2_shared[fso5[ty*bDx + tx]];
    //
    out[factor144*Idz + 3*facearr_factor + 0*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f3up[tx]] + seq2_shared[fso0[ty*bDx + tx]];
    out[factor144*Idz + 3*facearr_factor + 1*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f4up[tx]] + seq2_shared[fso1[ty*bDx + tx]];
    out[factor144*Idz + 3*facearr_factor + 2*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f5up[tx]] + seq2_shared[fso2[ty*bDx + tx]];
    out[factor144*Idz + 3*facearr_factor + 3*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f0up[tx]] + seq2_shared[fso3[ty*bDx + tx]];
    out[factor144*Idz + 3*facearr_factor + 4*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f1up[tx]] + seq2_shared[fso4[ty*bDx + tx]];
    out[factor144*Idz + 3*facearr_factor + 5*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f2up[tx]] + seq2_shared[fso5[ty*bDx + tx]];
    //
    out[factor144*Idz + 4*facearr_factor + 0*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f2up[tx]] + seq2_shared[fso0[ty*bDx + tx]];
    out[factor144*Idz + 4*facearr_factor + 1*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f3up[tx]] + seq2_shared[fso1[ty*bDx + tx]];
    out[factor144*Idz + 4*facearr_factor + 2*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f4up[tx]] + seq2_shared[fso2[ty*bDx + tx]];
    out[factor144*Idz + 4*facearr_factor + 3*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f5up[tx]] + seq2_shared[fso3[ty*bDx + tx]];
    out[factor144*Idz + 4*facearr_factor + 4*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f0up[tx]] + seq2_shared[fso4[ty*bDx + tx]];
    out[factor144*Idz + 4*facearr_factor + 5*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f1up[tx]] + seq2_shared[fso5[ty*bDx + tx]];
    //
    out[factor144*Idz + 5*facearr_factor + 0*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f1up[tx]] + seq2_shared[fso0[ty*bDx + tx]];
    out[factor144*Idz + 5*facearr_factor + 1*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f2up[tx]] + seq2_shared[fso1[ty*bDx + tx]];
    out[factor144*Idz + 5*facearr_factor + 2*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f3up[tx]] + seq2_shared[fso2[ty*bDx + tx]];
    out[factor144*Idz + 5*facearr_factor + 3*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f4up[tx]] + seq2_shared[fso3[ty*bDx + tx]];
    out[factor144*Idz + 5*facearr_factor + 4*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f5up[tx]] + seq2_shared[fso4[ty*bDx + tx]];
    out[factor144*Idz + 5*facearr_factor + 5*ORIENTATIONS*fsize + ty*fsize + local_Idx] = ALPHABET*seq1_shared[f0up[tx]] + seq2_shared[fso5[ty*bDx + tx]];
    //
    //
}




// __global__ void combine_spectra_skeleton_seqseq(float *spectra144, int *binding_spectra_ptrs,
//                                                 float *seqseq_ready_MJ, int n_pairs, int seq_len,
//                                                 int edge_len) {
//     //
//     // binding spectra array containing 144 binding energy levels "coordinates" (pairs of contacting residues),
//     // so that there are #_pairs spectras and each spectrum is an array of 144 energy levels (face-to-face and orientation)
//     // and in turn, each energy level is a edge*edge pairs of contacting residues compressed in one int, pair -> (int/seqlen,int%seqlen)
//     // int *binding_spectra
//     //
//     // seqseq array contains MJ energies of all potential residue contacts between seqA and seqB, they're stored in the following form:
//     // seqA[0] pairing with seqB[i],i=0..SeqLen first, seqA[1] pairing with seqB[i],i=0..SeqLen second, and so on and so forth ...
//     // float *seqseq_ready_MJ
//     //
//     // important!: we're expecting "binding_spectra" and "seqseq_ready_MJ" to have 1 to 1 correspondance (ordered the same way)...
//     //
//     // so, what we're about to do is to combine binding_spectra and seqseq to get 144-spectras for each of contacting sequences (folded in a cube)
//     // binding_spectra - is very very reminiscent of the ELL format for the sparse matrices - more on that later...
//     // in other words it is going to be a multiplication of #_pairs of sparse-matrix-vector instances: each matrix being
//     // (144 by edge*edge) in a sparse sense, or (144 by seq_len*seq_len) and a vector is dense and is of SeqLen*SeqLen length ...
//     //
//     // The result of this combination is going to be #_pairs of 144 float spectral vectors
//     // further processing of this 144 vectors (or look at them as row in a #_pairs by 144 matrix) implies min reduce (with index tracking)
//     // and also +reduce with prior exponentiation - P_nat calculation ...
//     //
//     // longest SeqLen supported is going to be 125, so max vector is 125*125=15625, so for floats that's 61KB!!!
//     // 4*4*4 results in 64*64*4 -> 16KB
//     // 3*3*3 results in 27*27*4 -> ~3KB
//     //
//     // let's do for 444 and 333 first, kinda saying that we have sufficient amount of shared memory ...
//     //
//     extern __shared__ float seqseq_evec[];
//     // program is designed to be 1 X-block wide, not any bigger ...
//     int tx = threadIdx.x;
//     int ty = threadIdx.y;
//     // int tz = threadIdx.z;
//     // int bx = blockIdx.x;
//     int by = blockIdx.y;
//     // int bz = blockIdx.z;
//     int bDx = blockDim.x;
//     int bDy = blockDim.y;
//     // int bDz = blockDim.z;
//     // int gDx = gridDim.x;
//     // int gDy = gridDim.y;
//     // int Idx = bx*bDx+tx;
//     int Idy = by*bDy+ty;
//     // int Idz = bz*bDz+tz;
//     //
//     // thread block through-way index for coalesced loading into shared memory ...
//     // from 0 to bDx*bDy-1, kinda like bDx*bDy elements typically ~256
//     int threads_in_block = bDx*bDy;
//     int tid = tx + ty*bDx;
//     //
//     int face_size = edge_len*edge_len;
//     int seq_squared = seq_len * seq_len;
//     // //
//     // int local_tx = min(face_size-1,tx);
//     // int this_pair_ptr = binding_spectra_ptrs[Idy*face_size + local_tx];
//     // //
//     // load binding spectra skeleton (kinda coalesced loading)...
//     int this_pair_ptr = (tx < face_size) ? binding_spectra_ptrs[Idy*face_size+tx] : seq_squared;
//     //
//     // loading into shared memory - coalesced in several iterations ~ seq_squared/threads_in_block
//     for (int i = 0; i < (seq_squared-1)/threads_in_block+1; i++) {
//         //
//         seqseq_evec[tid] = seqseq_ready_MJ[(Idy/144)*seq_squared+tid];
//         // when tid exceed the seq_squared (vec_width) it'll start over loading from 0
//         // a bit of wasting but better than warp divergence ...
//         tid = (tid + threads_in_block)%seq_squared;
//     }
//     // extra element for reduction smaller then the half-warp ...
//     seqseq_evec[seq_squared] = 0.0f;
//     //
//     __syncthreads();
//     // wait for the load to finish ...
//     //
//     // here is the point where we access shared mem in an unpredicted fashion ...
//     float e_pre_sum = seqseq_evec[this_pair_ptr];
//     //
//     // i'm not sure if we need to syncthreads at this point or not ...
//     // probably not because we're going to summ only within each warp ...
//     __syncthreads();
//     //
//     // now each of this registers "this_pair_MJ" contains the energy value to be summed  up ...
//     // using register shuffling within the warp - blazing fast!
//     e_pre_sum += __shfl_down(e_pre_sum,8,16);
//     e_pre_sum += __shfl_down(e_pre_sum,4,16);
//     e_pre_sum += __shfl_down(e_pre_sum,2,16);
//     e_pre_sum += __shfl_down(e_pre_sum,1,16);
//     // store result back to global ...
//     // reduction of the whole block resides in the 0th element ...
//     if ( tx==0 ) { spectra144[Idy] = e_pre_sum; }
//     //
// }


__global__ void combine_spectra_skeleton_seqseq(float *spectra144, int *binding_spectra_ptrs,
                                                float *seqseq_ready_MJ,
                                                // float *emin_out, int *mindex_out,
                                                int n_pairs, int seq_len,
                                                int edge_len) {
    //
    // binding spectra array containing 144 binding energy levels "coordinates" (pairs of contacting residues),
    // so that there are #_pairs spectras and each spectrum is an array of 144 energy levels (face-to-face and orientation)
    // and in turn, each energy level is a edge*edge pairs of contacting residues compressed in one int, pair -> (int/seqlen,int%seqlen)
    // int *binding_spectra
    //
    // seqseq array contains MJ energies of all potential residue contacts between seqA and seqB, they're stored in the following form:
    // seqA[0] pairing with seqB[i],i=0..SeqLen first, seqA[1] pairing with seqB[i],i=0..SeqLen second, and so on and so forth ...
    // float *seqseq_ready_MJ
    //
    // important!: we're expecting "binding_spectra" and "seqseq_ready_MJ" to have 1 to 1 correspondance (ordered the same way)...
    //
    // so, what we're about to do is to combine binding_spectra and seqseq to get 144-spectras for each of contacting sequences (folded in a cube)
    // binding_spectra - is very very reminiscent of the ELL format for the sparse matrices - more on that later...
    // in other words it is going to be a multiplication of #_pairs of sparse-matrix-vector instances: each matrix being
    // (144 by edge*edge) in a sparse sense, or (144 by seq_len*seq_len) and a vector is dense and is of SeqLen*SeqLen length ...
    //
    // The result of this combination is going to be #_pairs of 144 float spectral vectors
    // further processing of this 144 vectors (or look at them as row in a #_pairs by 144 matrix) implies min reduce (with index tracking)
    // and also +reduce with prior exponentiation - P_nat calculation ...
    //
    // longest SeqLen supported is going to be 125, so max vector is 125*125=15625, so for floats that's 61KB!!!
    // 4*4*4 results in 64*64*4 -> 16KB
    // 3*3*3 results in 27*27*4 -> ~3KB
    //
    // let's do for 444 and 333 first, kinda saying that we have sufficient amount of shared memory ...
    //
    // program is designed to be 1 X-block wide, not any bigger ...
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    //
    // int bx = blockIdx.x;
    int by = blockIdx.y;
    //
    int bDx = blockDim.x;
    int bDy = blockDim.y;
    //
    int Idy = by*bDy*9+ty;
    int pair_idy = Idy/144;
    //
    // thread block through-way index for coalesced loading into shared memory ...
    // from 0 to bDx*bDy-1, kinda like bDx*bDy elements typically ~256
    int threads_in_block = bDx*bDy;
    int tid = tx + ty*bDx;
    int tid_cpy = tid;
    //
    int face_size  = edge_len * edge_len;
    int seq_squared = seq_len * seq_len;
    //
    // shared memory alloc ...
    extern __shared__ float scratch[];
    //
    float *seqseq_evec = &scratch[0];
    // UNUSED AT THE MOMENT ...
    // float *spec_local = &scratch[seq_squared];
    //
    // // UNIVERSAL LOADING ...
    // // loading into shared memory - coalesced in several iterations ~ seq_squared/threads_in_block
    // for (int i = 0; i < (seq_squared-1)/threads_in_block+1; i++) {
    //     seqseq_evec[tid_cpy] = seqseq_ready_MJ[pair_idy*seq_squared+tid_cpy];
    //     // when tid_cpy exceed the seq_squared (vec_width) it'll start over loading from 0
    //     // a bit of wasting but better than warp divergence ...
    //     tid_cpy = (tid_cpy + threads_in_block)%seq_squared;
    // }
    // extra element for reduction smaller then the half-warp ...
    // seqseq_evec[seq_squared] = 0.0f;
    // // //  SUPER FAST LOADING FOR 16-MERS ONLY ...
    #pragma unroll
    // for (int i = 0; i < seq_squared/(4*threads_in_block); i++) {
    for (int i = 0; i < 4; i++) {
        // assert seq_squared%4 == 0
        reinterpret_cast<float4*>(seqseq_evec)[tid_cpy] = reinterpret_cast<float4*>(seqseq_ready_MJ)[pair_idy*(seq_squared>>2)+tid_cpy];
        tid_cpy += threads_in_block;
    }    
    __syncthreads();
    //
    //
    // assert 144~bDy, that's true for bDy=16 ...
    #pragma unroll
    // for (int i = 0; i < (144-1)/bDy+1; i++) {
    for (int i = 0; i < 9; i++) {
        // int this_pair_ptr = (tx < face_size) ? binding_spectra_ptrs[(Idy+i*bDy)*face_size+tx] : seq_squared;
        int this_pair_ptr = binding_spectra_ptrs[(Idy+i*bDy)*face_size+tx];
        //
        // here is the point where we access shared mem in an unpredicted fashion ...
        float e_pre_sum = seqseq_evec[this_pair_ptr];
        // i'm not sure if we need to syncthreads at this point or not ...
        // probably not because we're going to summ only within each warp ...
        //
        // now each of this registers "e_pre_sum" contains the energy value to be summed up ...
        // using register shuffling within the warp - blazing fast!
        e_pre_sum += __shfl_down(e_pre_sum,8,16);
        e_pre_sum += __shfl_down(e_pre_sum,4,16);
        e_pre_sum += __shfl_down(e_pre_sum,2,16);
        e_pre_sum += __shfl_down(e_pre_sum,1,16);
        // store result back to global ...
        // reduction of the whole block resides in the 0th element ...
        if ( tx==0 ) { spectra144[(Idy+i*bDy)] = e_pre_sum; }
    }
}



__global__ void combine_spectra_skeleton_seqseq_FF(float *spectra144, int *binding_spectra_ptrs, float* ForceField, int n_pairs, int seq_len, int edge_len) {
    //
    // binding spectra array containing 144 binding energy levels "coordinates" (pairs of contacting residues),
    // so that there are #_pairs spectras and each spectrum is an array of 144 energy levels (face-to-face and orientation)
    // and in turn, each energy level is a edge*edge pairs of contacting residues compressed in one int, pair -> (int/seqlen,int%seqlen)
    // int *binding_spectra
    //
    // seqseq array contains MJ energies of all potential residue contacts between seqA and seqB, they're stored in the following form:
    // seqA[0] pairing with seqB[i],i=0..SeqLen first, seqA[1] pairing with seqB[i],i=0..SeqLen second, and so on and so forth ...
    // float *seqseq_ready_MJ
    //
    // important!: we're expecting "binding_spectra" and "seqseq_ready_MJ" to have 1 to 1 correspondance (ordered the same way)...
    //
    // so, what we're about to do is to combine binding_spectra and seqseq to get 144-spectras for each of contacting sequences (folded in a cube)
    // binding_spectra - is very very reminiscent of the ELL format for the sparse matrices - more on that later...
    // in other words it is going to be a multiplication of #_pairs of sparse-matrix-vector instances: each matrix being
    // (144 by edge*edge) in a sparse sense, or (144 by seq_len*seq_len) and a vector is dense and is of SeqLen*SeqLen length ...
    //
    // The result of this combination is going to be #_pairs of 144 float spectral vectors
    // further processing of this 144 vectors (or look at them as row in a #_pairs by 144 matrix) implies min reduce (with index tracking)
    // and also +reduce with prior exponentiation - P_nat calculation ...
    //
    // longest SeqLen supported is going to be 125, so max vector is 125*125=15625, so for floats that's 61KB!!!
    // 4*4*4 results in 64*64*4 -> 16KB
    // 3*3*3 results in 27*27*4 -> ~3KB
    //
    // let's do for 444 and 333 first, kinda saying that we have sufficient amount of shared memory ...
    //
    // program is designed to be 1 X-block wide, not any bigger ...
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    //
    // int bx = blockIdx.x;
    int by = blockIdx.y;
    //
    int bDx = blockDim.x;
    int bDy = blockDim.y;
    //
    int Idy = by*bDy*9+ty;
    // int pair_idy = Idy/144;
    //
    // thread block through-way index for coalesced loading into shared memory ...
    // from 0 to bDx*bDy-1, kinda like bDx*bDy elements typically ~256
    // int threads_in_block = bDx*bDy;
    int tid = tx + ty*bDx;
    // int tid_cpy = tid;
    //
    //
    extern __shared__ float scratch[];
    //
    float *FF = &scratch[0];
    // Force fiels loading ...
    FF[tid] = ForceField[tid];
    int tid_my = min(184,tid); // up to 441 ...
    FF[256+tid_my] = ForceField[256+tid_my];
    __syncthreads();
    //
    int face_size  = edge_len * edge_len;
    // int seq_squared = seq_len * seq_len;
    //
    //
    // assert 144~bDy, that's true for bDy=16 ...
    #pragma unroll
    // for (int i = 0; i < (144-1)/bDy+1; i++) {
    for (int i = 0; i < 9; i++) {
        // int this_pair_ptr = (tx < face_size) ? binding_spectra_ptrs[(Idy+i*bDy)*face_size+tx] : seq_squared;
        int this_pair_ptr = binding_spectra_ptrs[(Idy+i*bDy)*face_size+tx];
        //
        // here is the point where we access shared mem in an unpredicted fashion ...
        float e_pre_sum = FF[this_pair_ptr];
        // float e_pre_sum = ForceField[this_pair_ptr];
        // i'm not sure if we need to syncthreads at this point or not ...
        // probably not because we're going to summ only within each warp ...
        //
        // now each of this registers "e_pre_sum" contains the energy value to be summed up ...
        // using register shuffling within the warp - blazing fast!
        e_pre_sum += __shfl_down(e_pre_sum,8,16);
        e_pre_sum += __shfl_down(e_pre_sum,4,16);
        e_pre_sum += __shfl_down(e_pre_sum,2,16);
        e_pre_sum += __shfl_down(e_pre_sum,1,16);
        // store result back to global ...
        // reduction of the whole block resides in the 0th element ...
        if ( tx==0 ) { spectra144[(Idy+i*bDy)] = e_pre_sum; }
    }
    //
}




__global__ void min_reduction2D_144(float * mat, float * out_tmp, int * str_idx_tmp, int row_seq){
    //
    // thread and block coordinates ...
    int tx = threadIdx.x; int ty = threadIdx.y;
    int by = blockIdx.y;
    int bDx = blockDim.x; int bDy = blockDim.y;
    //
    int col_str = 144;
    int Idx = tx;
    int Idy = min(bDy * by + ty,row_seq-1);
    //
    float just_load;
    float pre_accum = FLT_MAX;
    float pre_accum_tmp;
    int min_index=0;
    int min_index_tmp;
    //
    // massive coalsced loading ...
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        just_load = mat[Idy*col_str + Idx];
        if (pre_accum > just_load){ pre_accum = just_load; min_index = Idx; }
        Idx += bDx;
    }
    __syncthreads();
    // using register shuffling within the warp - blazing fast!
    pre_accum_tmp=__shfl_down(pre_accum,8,16);
    min_index_tmp=__shfl_down(min_index,8,16);
    if (pre_accum > pre_accum_tmp){ pre_accum=pre_accum_tmp; min_index=min_index_tmp;};
    pre_accum_tmp=__shfl_down(pre_accum,4,16);
    min_index_tmp=__shfl_down(min_index,4,16);
    if (pre_accum > pre_accum_tmp){ pre_accum=pre_accum_tmp; min_index=min_index_tmp;};
    pre_accum_tmp=__shfl_down(pre_accum,2,16);
    min_index_tmp=__shfl_down(min_index,2,16);
    if (pre_accum > pre_accum_tmp){ pre_accum=pre_accum_tmp; min_index=min_index_tmp;};
    pre_accum_tmp=__shfl_down(pre_accum,1,16);
    min_index_tmp=__shfl_down(min_index,1,16);
    if (pre_accum > pre_accum_tmp){ pre_accum=pre_accum_tmp; min_index=min_index_tmp;};
    //
    if (tx == 0) {
        out_tmp[Idy] = pre_accum;
        str_idx_tmp[Idy] = min_index;
    }
}





__global__ void sub_exp_reduction144(float * mat, float * vec, float * out, int row_seq, float coeff){
    //
    // thread and block coordinates ...
    int tx = threadIdx.x; int ty = threadIdx.y;
    int by = blockIdx.y;
    int bDx = blockDim.x; int bDy = blockDim.y;
    //
    int col_str = 144;
    int Idx = tx;
    int Idy = min(bDy * by + ty,row_seq-1);
    //
    float pre_sum = 0.0f;
    float just_load;
    float subtract = vec[Idy];
    //
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        just_load = mat[Idy*col_str + Idx];
        pre_sum = pre_sum + __expf(coeff*(just_load - subtract));
        Idx += bDx;
    }
    //
    __syncthreads();
    //
    // using register shuffling within the warp - blazing fast!
    pre_sum += __shfl_down(pre_sum,8,16);
    pre_sum += __shfl_down(pre_sum,4,16);
    pre_sum += __shfl_down(pre_sum,2,16);
    pre_sum += __shfl_down(pre_sum,1,16);
    // store result back to global ...
    // reduction of the whole block resides in the 0th element ...
    if (tx == 0) { out[Idy] = 1.f/pre_sum; }
    //
}



















