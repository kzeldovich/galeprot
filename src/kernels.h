#define REDUCTION_GRIDX 16

#define SPLIT(total,chunk) ((total)-1)/(chunk)+1

void Check_CUDA_Error(const char *message);

// folding kernels ...
__global__ void kernel_convert_seq_lut_v2( unsigned char *seqarray, float *seq_vector_E, float *ForceField, unsigned char *idx_lut, int MaxSeqLen, int MeanignConts, int n_seq);

__global__ void sub_exp_reduction(   float * mat, const float * vec, float * out, const int col_str, const int row_seq, const float coeff);

__global__ void reduction_multi_blocks(float * input, float * out, int * out_idx, int col_str, int row_seq);

__global__ void reduction_one_block(float * input, int * input_idx, float * out, int * out_idx, int col_str, int row_seq);

__global__ void kernel_invert_Z_deltaG(float *Z, float *deltaG, int len, float boltzmann);



// for binding purposes only ...

__global__ void get_binding_spectra_coords_FF(int *out, unsigned char *faces_array, unsigned char *seqarray, int *order_conf, int *order_seq, int n_pairs, int edgelen, int seq_len);

__global__ void combine_spectra_skeleton_seqseq_FF(float *spectra144, int *binding_spectra_ptrs, float* ForceField, int n_pairs, int seq_len, int edge_len);

__global__ void min_reduction2D_144(float * mat, float * out_tmp, int * str_idx_tmp, int row_seq);

__global__ void sub_exp_reduction144(float * mat, float * vec, float * out_pnat, float * out_deltaG, int row_seq, float coeff);
