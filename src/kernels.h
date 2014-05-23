#define WORK_str 32

#define SPLIT(total,chunk) ((total)-1)/(chunk)+1

void Check_CUDA_Error(const char *message);


__device__ void lock(int *pmutex);


__device__ void unlock(int *pmutex);


__global__ void kernel_convert_seq_lut_v2( unsigned char *seqarray, float *seq_vector_E, float *ForceField, unsigned char *idx_lut, int MaxSeqLen, int MeanignConts, int n_seq);


__global__ void sub_exp_reduction(   float * mat, const float * vec, float * out, const int col_str, const int row_seq, const float coeff);


__global__ void experimental_reduction2D(int *mutexes, float * mat, float * out, int * out_idx, int col_str, int row_seq);


//not in use
__global__ void min_reduction2D_unified(float * mat, float * out_tmp, int * str_idx_tmp, int col_str, int row_seq, int col_tmp);


//not in use
__global__ void min_reduction2D_oneblockx(float * in_tmp, float * out, int * str_idx_tmp, int * str_idx, int row_seq, int col_tmp);


__global__ void kernel_invert_Z(float *Z, int len);



// for binding purposes only ...
// __global__ void sort_and_copy_sequences(unsigned char *out, unsigned char *in, int *order, int len, int seq_len);


__global__ void get_seq_pairwise_vector(unsigned char *seqarray, float *seqseq_vector_E, int *order, float *ForceField, int n_pairs, int seq_len);


__global__ void get_binding_spectra_coords(int *out, unsigned char *faces_array, int *order, int n_pairs, int edgelen, int seq_len);


__global__ void get_binding_spectra_coords_FF(int *out, unsigned char *faces_array, unsigned char *seqarray, int *order_conf, int *order_seq, int n_pairs, int edgelen, int seq_len);


__global__ void combine_spectra_skeleton_seqseq(float *spectra144, int *binding_spectra_ptrs, float *seqseq_ready_MJ, int n_pairs, int seq_len, int edge_len);


__global__ void combine_spectra_skeleton_seqseq_FF(float *spectra144, int *binding_spectra_ptrs, float* ForceField, int n_pairs, int seq_len, int edge_len);


__global__ void min_reduction2D_144(float * mat, float * out_tmp, int * str_idx_tmp, int row_seq);


__global__ void sub_exp_reduction144(float * mat, float * vec, float * out, int row_seq, float coeff);
