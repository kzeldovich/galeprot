      
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




// reduction with exp
__global__ void sub_exp_reduction(  float * mat,
                                    const float * vec,
                                    float * out,
                                    const int col_str,
                                    const int row_seq,
                                    const float coeff){
    // thread and block coordinates ...
    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x; int by = blockIdx.y;
    int bDx = blockDim.x; int bDy = blockDim.y;
    int gDx = gridDim.x; //int gDy = gridDim.y;
    //
    int Idx = bDx * bx + tx;
    int Idy = bDy * by + ty;
    int gridSize = gDx * bDx;
    // exit if y -dim is out of bound ...
    if (Idy>=row_seq) {return; }
    //
    float subtract = vec[Idy];
    //
    extern __shared__ float pre_result[];
    // initialize pre_result ...
    pre_result[bDx*ty+tx] = 0.0f;
    //
    while (Idx < col_str) {
        float loaded1 = mat[Idy*col_str + Idx];
        pre_result[bDx*ty+tx] += __expf(coeff*(loaded1 - subtract));
        Idx += gridSize;
    }
    //
    __syncthreads();
    //
    if (tx<8) pre_result[bDx*ty+tx] += pre_result[bDx*ty+tx+8]; __syncthreads();
    if (tx<4) pre_result[bDx*ty+tx] += pre_result[bDx*ty+tx+4]; __syncthreads();
    if (tx<2) pre_result[bDx*ty+tx] += pre_result[bDx*ty+tx+2]; __syncthreads();
    if (tx<1) pre_result[bDx*ty+tx] += pre_result[bDx*ty+tx+1]; __syncthreads();
    //
    // store result back to global ...
    // reduction of the whole block resides in the 0th element ...
    if (tx == 0) { atomicAdd(&out[Idy], pre_result[bDx*ty+tx]); }
    //
}




__global__ void reduction_multi_blocks(float * input, float * out, int * out_idx, int col_str, int row_seq){
    // thread and block coordinates ...
    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x; int by = blockIdx.y;
    int bDx = blockDim.x; int bDy = blockDim.y;
    int gDx = gridDim.x;
    //
    int Idx = bDx*bx + tx;
    int Idy = bDy*by + ty;
    int gridSize = gDx*bDx;
    // exit thread if the row counter is out of bounds ...
    if (Idy >= row_seq) { return; }
    //
    extern __shared__ float scratch[];
    // initialize pre_result ...
    volatile float *pre_result = &scratch[0];
    volatile int   *min_result = (int *)&scratch[bDx*bDy];
    // init shared mem arrays ...
    pre_result[bDx*ty+tx] = FLT_MAX;
    min_result[bDx*ty+tx] = 0;
    // coalesced loading ...
    while (Idx < col_str) {
        float loaded1 = input[Idy*col_str + Idx];
        if (pre_result[bDx*ty + tx] > loaded1){ pre_result[bDx*ty + tx] = loaded1; min_result[bDx*ty + tx] = Idx; } 
        Idx += gridSize;
    }
    //
    __syncthreads();
    //
    if (tx<8) {
        if (pre_result[bDx*ty+tx] > pre_result[bDx*ty+tx+8]){
            pre_result[bDx*ty+tx] = pre_result[bDx*ty+tx+8];
            min_result[bDx*ty+tx] = min_result[bDx*ty+tx+8];
        }
    }; __syncthreads();
    if (tx<4) {
        if (pre_result[bDx*ty+tx] > pre_result[bDx*ty+tx+4]){
            pre_result[bDx*ty+tx] = pre_result[bDx*ty+tx+4];
            min_result[bDx*ty+tx] = min_result[bDx*ty+tx+4];
        }
    }; __syncthreads();
    if (tx<2) {
        if (pre_result[bDx*ty+tx] > pre_result[bDx*ty+tx+2]){
            pre_result[bDx*ty+tx] = pre_result[bDx*ty+tx+2];
            min_result[bDx*ty+tx] = min_result[bDx*ty+tx+2];
        }
    }; __syncthreads();
    if (tx<1) {
        if (pre_result[bDx*ty+tx] > pre_result[bDx*ty+tx+1]){
            pre_result[bDx*ty+tx] = pre_result[bDx*ty+tx+1];
            min_result[bDx*ty+tx] = min_result[bDx*ty+tx+1];
        }
    }; __syncthreads();
    //
    if (tx == 0) {
        out[Idy*gDx + bx] = pre_result[bDx*ty];
        out_idx[Idy*gDx + bx] = min_result[bDx*ty];
    }
}




__global__ void reduction_one_block(float * input, int * input_idx, float * out, int * out_idx, int col_str, int row_seq){
    // thread and block coordinates ...
    int tx = threadIdx.x; int ty = threadIdx.y;
    int by = blockIdx.y;
    int bDx = blockDim.x; int bDy = blockDim.y;
    // one block implies:
    // bx = 0; gDx = 1;
    int Idx = tx;
    int Idy = bDy * by + ty;
    int gridSize = bDx;
    // exit thread if the row counter is out of bounds ...
    if (Idy >= row_seq) {return; }
    //
    extern __shared__ float scratch[];
    // initialize pre_result ...
    volatile float *pre_result = &scratch[0];
    volatile int   *min_result = (int *)&scratch[bDx*bDy];
    // init shared mem arrays ...
    pre_result[bDx*ty+tx] = FLT_MAX;
    min_result[bDx*ty+tx] = 0;
    //
    // coalesced loading ...
    while (Idx < col_str) {
        float loaded1 = input[Idy*col_str + Idx];
        if (pre_result[bDx*ty + tx] > loaded1){ pre_result[bDx*ty + tx] = loaded1; min_result[bDx*ty + tx] = input_idx[Idy*col_str + Idx]; } 
        Idx += gridSize;
    }
    __syncthreads();
    //
    //
    if (tx<8) {
        if (pre_result[bDx*ty+tx] > pre_result[bDx*ty+tx+8]){
            pre_result[bDx*ty+tx] = pre_result[bDx*ty+tx+8];
            min_result[bDx*ty+tx] = min_result[bDx*ty+tx+8];
        }
    }; __syncthreads();
    if (tx<4) {
        if (pre_result[bDx*ty+tx] > pre_result[bDx*ty+tx+4]){
            pre_result[bDx*ty+tx] = pre_result[bDx*ty+tx+4];
            min_result[bDx*ty+tx] = min_result[bDx*ty+tx+4];
        }
    }; __syncthreads();
    if (tx<2) {
        if (pre_result[bDx*ty+tx] > pre_result[bDx*ty+tx+2]){
            pre_result[bDx*ty+tx] = pre_result[bDx*ty+tx+2];
            min_result[bDx*ty+tx] = min_result[bDx*ty+tx+2];
        }
    }; __syncthreads();
    if (tx<1) {
        if (pre_result[bDx*ty+tx] > pre_result[bDx*ty+tx+1]){
            pre_result[bDx*ty+tx] = pre_result[bDx*ty+tx+1];
            min_result[bDx*ty+tx] = min_result[bDx*ty+tx+1];
        }
    }; __syncthreads();
    //
    if (tx == 0) {
        out[Idy] = pre_result[bDx*ty];
        out_idx[Idy] = min_result[bDx*ty];
    }
    //
}




__global__ void kernel_invert_Z_deltaG(float *Z, float *deltaG, int len, float boltzmann){
    //
    int bDx = blockDim.x;
    int Idx = blockIdx.x*bDx + threadIdx.x  ;
    int gridSize = gridDim.x*bDx;
    //
    while (Idx < len) {
        // loading input ...
        float pnat = 1/Z[Idx];
        // storing pnat and dG ...
        deltaG[Idx] = -boltzmann*logf(pnat/(1-pnat));
        Z[Idx] = pnat;        
        Idx += gridSize;
    }

}





// binding kernels ...
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



// binding kernel 64 mers only! ...
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
    // C144/bDx == 9
    int Idy = by*bDy*(C144/bDx)+ty;
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
    float *pre_result = &scratch[0];
    float *FF = &scratch[bDx*bDy];
    // init ...
    pre_result[bDx*ty+tx] = 0.0f;
    // Force fiels loading - all 21*21 elements ...
    // first 256 elements ...
    FF[tid] = ForceField[tid];
    // last ALPHABET*ALPHABET==21*21==441 except for 256 items ...
    int tid_my = min(ALPHABET*ALPHABET-1-bDx*bDy,tid); // up to 441 ...
    FF[bDx*bDy+tid_my] = ForceField[bDx*bDy+tid_my];
    __syncthreads();
    //
    int face_size  = edge_len * edge_len;
    // assert 144~bDy, that's true for bDy=16 ...
    #pragma unroll
    // C144/bDx == 9
    for (int i = 0; i < 9; i++) {
        int this_pair_ptr = binding_spectra_ptrs[(Idy+i*bDy)*face_size+tx];
        // here is the point where we access shared mem in an unpredicted fashion ...
        pre_result[bDx*ty+tx] = FF[this_pair_ptr]; __syncthreads();
        //
        if (tx<8) pre_result[bDx*ty+tx] += pre_result[bDx*ty+tx+8]; __syncthreads();
        if (tx<4) pre_result[bDx*ty+tx] += pre_result[bDx*ty+tx+4]; __syncthreads();
        if (tx<2) pre_result[bDx*ty+tx] += pre_result[bDx*ty+tx+2]; __syncthreads();
        if (tx<1) pre_result[bDx*ty+tx] += pre_result[bDx*ty+tx+1]; __syncthreads();
        // store result back to global ...
        // reduction of the whole block resides in the 0th element ...
        if ( tx==0 ) { spectra144[(Idy+i*bDy)] = pre_result[bDx*ty+tx]; }
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
    int col_str = C144;
    int Idx = tx;
    int Idy = min(bDy * by + ty,row_seq-1);
    //
    float just_load;
    //
    extern __shared__ float scratch[];
    float *pre_result = &scratch[0];
    int *min_result = (int *)&scratch[bDx*bDy];
    // init ...
    pre_result[bDx*ty+tx] = FLT_MAX;
    min_result[bDx*ty+tx] = 0;
    //
    // massive coalesced loading ...
    #pragma unroll
    // C144/bDx == 9
    for (int i = 0; i < 9; i++) {
        just_load = mat[Idy*col_str + Idx];
        if (pre_result[bDx*ty+tx] > just_load){ pre_result[bDx*ty+tx] = just_load; min_result[bDx*ty+tx] = Idx; }
        Idx += bDx;
    }
    __syncthreads();
    //
    if (tx<8){
    		if (pre_result[bDx*ty+tx]>pre_result[bDx*ty+tx+8]){
    			pre_result[bDx*ty+tx] = pre_result[bDx*ty+tx+8];
    			min_result[bDx*ty+tx] = min_result[bDx*ty+tx+8];
    		}
    }; __syncthreads();
    if (tx<4){
    		if (pre_result[bDx*ty+tx]>pre_result[bDx*ty+tx+4]){
    			pre_result[bDx*ty+tx] = pre_result[bDx*ty+tx+4];
    			min_result[bDx*ty+tx] = min_result[bDx*ty+tx+4];
    		}
    }; __syncthreads();
    if (tx<2){
    		if (pre_result[bDx*ty+tx]>pre_result[bDx*ty+tx+2]){
    			pre_result[bDx*ty+tx] = pre_result[bDx*ty+tx+2];
    			min_result[bDx*ty+tx] = min_result[bDx*ty+tx+2];
    		}
    }; __syncthreads();
    if (tx<1){
    		if (pre_result[bDx*ty+tx]>pre_result[bDx*ty+tx+1]){
    			pre_result[bDx*ty+tx] = pre_result[bDx*ty+tx+1];
    			min_result[bDx*ty+tx] = min_result[bDx*ty+tx+1];
    		}
    }; __syncthreads();
    //
    if (tx == 0) {
        out_tmp[Idy] = pre_result[bDx*ty+tx];
        str_idx_tmp[Idy] = min_result[bDx*ty+tx];
    }
}





__global__ void sub_exp_reduction144(float * mat, float * vec, float * out_pnat, float * out_deltaG, int row_seq, float coeff){
    //
    // thread and block coordinates ...
    int tx = threadIdx.x; int ty = threadIdx.y;
    int by = blockIdx.y;
    int bDx = blockDim.x; int bDy = blockDim.y;
    //
    int col_str = C144;
    int Idx = tx;
    int Idy = min(bDy * by + ty,row_seq-1);
    //
    float just_load;
    float subtract = vec[Idy];
    //
    extern __shared__ float pre_result[]; // bDx * bDy
    pre_result[bDx*ty+tx] = 0.0f;
    //
    #pragma unroll
    // C144/bDx == 9 
    for (int i = 0; i < 9; i++) {
        just_load = mat[Idy*col_str + Idx];
        pre_result[bDx*ty+tx] = pre_result[bDx*ty+tx] + __expf(coeff*(just_load - subtract));
        Idx += bDx;
    }
    //
    __syncthreads();
    //
    if (tx<8) pre_result[bDx*ty+tx] += pre_result[bDx*ty+tx+8]; __syncthreads();
    if (tx<4) pre_result[bDx*ty+tx] += pre_result[bDx*ty+tx+4]; __syncthreads();
    if (tx<2) pre_result[bDx*ty+tx] += pre_result[bDx*ty+tx+2]; __syncthreads();
    if (tx<1) pre_result[bDx*ty+tx] += pre_result[bDx*ty+tx+1]; __syncthreads();
    // store result back to global ...
    // reduction of the whole block resides in the 0th element ...
    //
    //
    if (tx == 0) {
        // becuase coeff = -1/k_BT ...
        float pnat = 1.f/pre_result[bDx*ty+tx];
        float dG = (1.f/coeff)*logf(pnat/(1-pnat));
        out_pnat[Idy] = pnat;
        out_deltaG[Idy] = dG;
    }
    //
}



















