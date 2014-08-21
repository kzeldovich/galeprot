#include<math.h>
#include<stdio.h>
#include<time.h>
#include<stdlib.h>

#include<string.h>
#include<assert.h>

#include "galelib.h"

char aalist[]  = "CMFILVWYAGTSNQDEHRKPZ";

void MutateSequences(unsigned char *sequences, int seqlen, int num_seq){
    for (int i = 0; i < num_seq; i++){
        int residue_to_mutate = rand() % seqlen; //random int in range 0..seqlen-1
        unsigned char new_aa;
        do {
            new_aa = rand() % 20;
        } while (sequences[i*seqlen + residue_to_mutate] == new_aa);
        sequences[i*seqlen + residue_to_mutate] = new_aa;
    }
}

float average(float *arr, int len){
    //
    float avg = 0.0f;
    for (int i = 0; i < len; i++) { avg += arr[i]; }
    return avg/(float)len;
}

void MJtoLetterAA(char *out, unsigned char *in, int seqlen) {
    char aaorder[] = "CMFILVWYAGTSNQDEHRKPZ";
    for(int i = 0; i < seqlen; i++) {
        out[i] = (in[i]<=20 && in[i]>=0) ? aaorder[in[i]] : '-';
    }
}

void ComparePnatSwapSeq(unsigned char *seq_wt,unsigned char *seq_mut, float *pnat_wt, float *pnat_mut, int seqlen, int num_seq ){
    
    float T_design = 1.0e-4; // 1e-3 for 27/64
    
    for (int i = 0; i < num_seq; i++) {
        if (pnat_mut[i]>pnat_wt[i]) { // pnat increased = always accept new sequence
            memcpy(seq_wt + i*seqlen, seq_mut + i*seqlen, seqlen*sizeof(unsigned char));
            pnat_wt[i] = pnat_mut[i];
        } else { // pnat did not increase, play Monte Carlo
            float r = ((float)random())/RAND_MAX;
            if (r < exp((pnat_mut[i] - pnat_wt[i])/T_design)) {
                memcpy(seq_wt + i*seqlen,seq_mut + i*seqlen, seqlen*sizeof(unsigned char));
                pnat_wt[i] = pnat_mut[i];
            }
        }
    }
}


void PrintOutput(char *fn, unsigned char *seqarray, float *pnat, int num_seq, int seqlen)  {
    FILE *fps = fopen(fn, "w");
    for (int seq = 0; seq < num_seq; seq++) {
         fprintf(fps, "%.4f ", pnat[seq]);
         for (int res = 0; res < seqlen; res++) {
             fprintf(fps, "%c", aalist[seqarray[seq*seqlen + res]]);
             }
       fprintf(fps, "\n");
       }
    fclose(fps);
}                                                            

// main code here
int main(int argc, char **argv) {

    int fold_wsize, seqlen, i, num_conf, num_seq;
    float T;

    seqlen = 64;
    num_conf = 10000; // needed to load contact maps

    T=2; // temperature for Pnat calculation

    int num_iterations = 10; // iterations of design (one mutation per protein per iteration)

    num_seq = 100000;    // number of sequences to design
    fold_wsize = 20000;  // fold 20k sequences at a time

    // seed the random numbers
    srand(12345);

    // allocate GALE_Data
    GALE_Data *data_ptr;
    gale_init(&data_ptr, 0);

    // allocate sequence arrays
    unsigned char *seqarray =     (unsigned char *)malloc(seqlen*num_seq*sizeof(unsigned char)); assert(seqarray != NULL);
    unsigned char *seqarray_mut = (unsigned char *)malloc(seqlen*num_seq*sizeof(unsigned char)); assert(seqarray_mut != NULL);

    // Pnat
    float   *pnat = (float *)malloc(num_seq*sizeof(float));         assert(pnat != NULL);
    float   *pnat_mut = (float *)malloc(num_seq*sizeof(float));     assert(pnat != NULL);
    // index of native conformation
    int     *inat = (int *)malloc(num_seq*sizeof(int));             assert(inat != NULL);

    // load structural info, faces info & force field ...
    printf("Loading force field, structures\n");
    gale_load_forcefield(data_ptr, "../data/MJ96_letters.d");
    gale_load_contactmap(data_ptr, "../data/c-10k-64.d", num_conf);
    printf("loading complete!\n");

    // define worksize (this must precede all other "set" type of functions)...
    gale_set_fold_worksize(data_ptr, fold_wsize);

    // connect seqarray to the device
    gale_set_seqarray(data_ptr, seqarray, seqlen, num_seq);
    gale_set_pfold(data_ptr, pnat);
    gale_set_ifold(data_ptr, inat);

    // prepare ...
    gale_fold_prepare(data_ptr);
    
    // initialize random sequences
    for(i=0;i<num_seq*seqlen;i++) seqarray[i] = rand() % 20;
     
    gale_fold_compute(data_ptr, T);

    PrintOutput("random-sequences.txt", seqarray, pnat, num_seq, seqlen);
        
    // then set working seq and pnat to MUTANT ones 
    gale_set_seqarray(data_ptr, seqarray_mut, seqlen, num_seq);
    gale_set_pfold(data_ptr, pnat_mut);

    // iterate  design
    for ( i = 0; i < num_iterations; i++) {
        memcpy(seqarray_mut, seqarray, seqlen*num_seq*sizeof(unsigned char));
        MutateSequences(seqarray_mut, seqlen, num_seq);

        gale_fold_compute(data_ptr, T); // folds seqarray_mut, output goes to pnat_mut
        
        ComparePnatSwapSeq( seqarray, seqarray_mut, pnat, pnat_mut, seqlen, num_seq );
        float avg_p = average(pnat, num_seq);
        printf("Step %d/%d  avg Pnat %.4f \n", i, num_iterations, avg_p);
    }

    // fold designed sequences again and print them out
    gale_set_seqarray(data_ptr, seqarray, seqlen, num_seq);
    gale_set_pfold(data_ptr, pnat);
    gale_set_ifold(data_ptr, inat);
    gale_fold_compute(data_ptr, T);    
    PrintOutput("designed-sequences.txt", seqarray, pnat, num_seq, seqlen);

    if (seqarray != NULL) { free(seqarray); }
    if (pnat != NULL) { free(pnat); }
    if (inat != NULL) { free(inat); }
    gale_close(&data_ptr);
    //
    return 0;
}

