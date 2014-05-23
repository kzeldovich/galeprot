
/*  GPU Accelerated Latice Evolution  */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <float.h>
#include <limits.h>
#include <assert.h>
#include <stddef.h>

#define ALPHABET 21
#define FACES 6
#define ORIENTATIONS 4
#define C144 144 // FACES*FACES * ORIENTATIONS


#define GALE_RETURN_PNAT_FOLD 0x1
#define GALE_RETURN_INDEX_FOLD 0x2
#define GALE_RETURN_DG_FOLD 0x4
#define GALE_RETURN_EMIN_FOLD 0x8
#define GALE_RETURN_SPECTRUM_FOLD 0x10

#define GALE_RETURN_PNAT_BIND 0x1
#define GALE_RETURN_INDEX_BIND 0x2
#define GALE_RETURN_DG_BIND 0x4
#define GALE_RETURN_EMIN_BIND 0x8
#define GALE_RETURN_SPECTRUM_BIND 0x10

#define GALE_REPORTS_QUIET 0
#define GALE_REPORTS_DEFAULT 1
#define GALE_REPORTS_VERBOSE 2
#define GALE_REPORTS_DEBUG 3


typedef struct {
    // service variables ...
    int outlist_fold;
    int outlist_bind;
    int report_level;

    // all the data for host-device interfacing ...
    
    // HOST some host arrays to be ommited later on ...
    // general & folding related ...
    unsigned char *contactmap_host;
    float *forcefield_host;
    unsigned char *seqarray_host;
    float *pfold_host; 
    float *gfold_host; 
    int   *ifold_host;
    float *efold_host; 
    float *sfold_host; 
    // binding related ...
    int *bind_seq_pairs_host;
    int *bind_conf_pairs_host;
    unsigned char *bind_faces_host; //binding faces in all orientations ...
    float *pbind_host; 
    float *gbind_host; 
    int   *ibind_host;
    float *ebind_host; 
    float *sbind_host; 

    //DEVICE
    // general & folding related ...
    unsigned char *seqarray_device;
    unsigned char *idxlut_device;
    float *contactmap_matrix_device;
    float *forcefield_device;
    float *seq_evector_device;
    float *pfold_device;
    float *gfold_device; 
    int   *ifold_device;
    float *efold_device; 
    float *sfold_device;
    // binding related ...
    int *bind_seq_pairs_device;
    int *bind_conf_pairs_device;
    unsigned char *bind_faces_device; //binding faces in all orientations ...
    // float *seqseq_evectors_device;
    int *bind_residues_device;
    float *pbind_device;
    float *gbind_device; 
    int   *ibind_device;
    float *ebind_device; 
    float *sbind_device;

    // VARIABLES ...
    int max_seqlen;
    int max_numcontacts;
    int existing_contacts;
    int num_seq;
    int num_conf;
    // binding only ...
    int num_conf_bind; // this should be equal to num_conf ... but who knows ...
    int edge_len;
    int face_size; // edge^2
    int num_pairs;
    // chunks ...
    int fold_chunk_size;
    int bind_chunk_size;
    //
} GALE_Data;

void gale_init(GALE_Data **gd, int deviceId);
////////////////////////
void gale_load_forcefield(GALE_Data *gd, const char *fname);
void gale_load_contactmap(GALE_Data *gd, const char *fname, int num_conf);
void gale_load_bind_faces(GALE_Data *gd, const char *fname, int num_conf, int edge_len);
////////////////////////
void gale_set_fold_worksize(GALE_Data *gd, int fold_worksize);
void gale_set_bind_worksize(GALE_Data *gd, int bind_worksize);
void gale_set_report_level(GALE_Data *gd, int report_level);
void gale_set_output_fold(GALE_Data *gd, int fold_output_list);
void gale_set_output_bind(GALE_Data *gd, int bind_output_list);
////////////////////////
void gale_set_seqarray(GALE_Data *gd, unsigned char *seqarray, int seqlen, int num_seq);
void gale_set_bind_pairs(GALE_Data *gd, int *seq_pairs, int *conf_pairs, int num_pairs);
////////////////////////
void gale_set_pfold(GALE_Data *gd, float *pfold_array);
void gale_set_gfold(GALE_Data *gd, float *gfold_array);
void gale_set_ifold(GALE_Data *gd, int *ifold_array);
void gale_set_efold(GALE_Data *gd, float *efold_array);
void gale_set_sfold(GALE_Data *gd, float *sfold_array);
////////////////////////
void gale_set_pbind(GALE_Data *gd, float *pbind_array);
void gale_set_gbind(GALE_Data *gd, float *gbind_array);
void gale_set_ibind(GALE_Data *gd, int *ibind_array);
void gale_set_ebind(GALE_Data *gd, float *ebind_array);
void gale_set_sbind(GALE_Data *gd, float *sbind_array);
////////////////////////
void gale_fold_prepare(GALE_Data *gd);
void gale_bind_prepare(GALE_Data *gd);
void gale_fold_unprepare(GALE_Data *gd);
void gale_bind_unprepare(GALE_Data *gd);
////////////////////////
void gale_fold_compute(GALE_Data *gd, float fold_temp);
void gale_bind_compute(GALE_Data *gd, float bind_temp);
////////////////////////
void gale_close(GALE_Data **gd);




