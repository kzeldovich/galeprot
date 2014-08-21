#include<math.h>
#include<stdio.h>
#include<time.h>
#include<stdlib.h>
#include "galelib.h"
//
void GenerateSequences(unsigned char **sequences, int seqlen, int num_seq);
void MutateSequences(unsigned char *sequences, int seqlen, int num_seq);
float average(float *arr, int len);
void MJtoLetterAA(char *out, unsigned char *in, int seqlen);

void ComparePnatSwapSeq(unsigned char *seq_wt,unsigned char *seq_mut, float *pnat_wt, float *pnat_mut, int seqlen, int num_seq ){
    //
    float T_design = 1.0e-4; // 1e-3 for 27/64
    //
    for (int i = 0; i < num_seq; i++) {
        if (pnat_mut[i]>pnat_wt[i]) { // pnat increased ...
            memcpy(seq_wt + i*seqlen,seq_mut + i*seqlen, seqlen*sizeof(unsigned char));
            pnat_wt[i] = pnat_mut[i];
        } else { // pnat didnt grow ...
            float r = ((float)random())/RAND_MAX;
            if (r < exp((pnat_mut[i] - pnat_wt[i])/T_design)) {
                memcpy(seq_wt + i*seqlen,seq_mut + i*seqlen, seqlen*sizeof(unsigned char));
                pnat_wt[i] = pnat_mut[i];
            }
        }
    } // next sequence ...
}


void get_cost(unsigned char *seq, int seqlen, int num_seq, float *pnat, float *metab_cost, float weight) 
{
int i;
float cost;

for(int k=0;k<num_seq;k++){

cost =0;
for(int m=0;m<seqlen;m++) {

cost += metab_cost[seq[k*seqlen+m]];

}

pnat[k] -= weight*cost;

}

}


void read_metab_cost(float *c, char *fn)
{
int i,j;
float x;
char a;
char aalist[] = "CMFILVWYAGTSNQDEHRKP";
for(i=0;i<20;i++) c[i]=0.;
FILE *f = fopen(fn,"r");
for(i=0;i<20;i++) {
fscanf(f,"%c %f\n", &a, &x);

printf("%c: \n",a);
for(j=0;j<20;j++) if (aalist[j]==a) { c[j]=x; printf("found\n"); }

}
fclose(f);

for(i=0;i<20;i++) printf("%d %c %.1f\n",i, aalist[i], c[i]);
}


//
//
int main(int argc, char **argv) {
    //
    int fold_wsize;
    int seqlen;
    char basename[100];
    float weight;
    //
    char f_field_fname[] = "MJ96.d";
    char str_file_name[100]; 
    
    seqlen = atoi(argv[1]);
    weight = atof(argv[2]);

    srand(atoi(argv[3]));

    int NNN = seqlen * 50;
    
    sprintf(str_file_name,"c-10k-%d.d",seqlen);
    fold_wsize = 20000;

    float metab_cost[20];
    int metab_idx[20];
    read_metab_cost(metab_cost, "akashi-cost.d");
    int ii; float s=0;
    for(ii=0;ii<20;ii++) s += metab_cost[ii]*metab_cost[ii];
    for(ii=0;ii<20;ii++) metab_cost[ii] /=sqrt(s);
    for(ii=0;ii<20;ii++) metab_idx[ii]=ii;

    // allocate some GALE struct ...
    GALE_Data *data_ptr;
    gale_init_null(&data_ptr);
    //
    unsigned char *seqarray;
    int num_seq = 100000;
    GenerateSequences( &seqarray, seqlen, num_seq);
    // allocate pnat & inat arrays, binding array to be allocated too ...
    unsigned char *seqarray_mut = (unsigned char *)malloc(seqlen*num_seq*sizeof(unsigned char)); assert(seqarray_mut != NULL);
    float   *pnat = (float *)malloc(num_seq*sizeof(float));         assert(pnat != NULL);
    float   *pnat_mut = (float *)malloc(num_seq*sizeof(float));     assert(pnat != NULL);
    int     *inat = (int *)malloc(num_seq*sizeof(int));             assert(inat != NULL);
    //
    // load structural info, faces info & force field ...
    printf("\nLOADING PROCEDURES ...\n");
    gale_load_forcefield(data_ptr, f_field_fname);
    gale_load_contactmap(data_ptr, str_file_name);
    printf("LOADING COMPLETE ...\n");
    //
    // define worksize (this must precede all other "set" type of functions)...
    printf("\nSET CHUNKS ...\n");

    gale_set_fold_worksize(data_ptr, fold_wsize);
    printf("CHUNKS ARE SET ...\n");
    //
    // connect seqarray to the device ...
    printf("\nMORE TO SET...\n");
    gale_set_seqarray(data_ptr, seqarray, seqlen, num_seq);
    gale_set_pfold(data_ptr, pnat);
    gale_set_ifold(data_ptr, inat);
    // prepare ...
    gale_fold_prepare(data_ptr);
    


    float T;
//    for(T=0.3;T<1.5;T+=0.2) 
//    for(T=0.2;T<6.01;T+=0.1)
    T =3;
    int shuffle;
    for(shuffle=0; shuffle<100; shuffle++)
    {

    if (shuffle>0)
    {
    int ii; float f1, f2; int d1,d2;
    int idx1, idx2;
    for(ii=0; ii<100; ii++) {
           idx1 = rand() % 20;
           idx2 = rand() % 20;
           f1 = metab_cost[idx1];
           f2 = metab_cost[idx2];
           metab_cost[idx2]=f1;
           metab_cost[idx1]=f2;
           d1=metab_idx[idx1];
           d2=metab_idx[idx2];
           metab_idx[idx2]=d1;
           metab_idx[idx1]=d2;
           }
    
    }

    for(weight=0; weight<0.1; weight+=0.005) {

    sprintf(basename, "L%d_w%.3f_T%.2f_S%d",seqlen,weight,T,shuffle);

    int i;
    for(i=0;i<num_seq*seqlen;i++) seqarray[i] = rand() % 20;

    gale_set_pfold(data_ptr, pnat);
    gale_set_ifold(data_ptr, inat);
     
    gale_fold_compute(data_ptr, T);
    get_cost(seqarray, seqlen, num_seq, pnat, metab_cost, weight);
    
    
    // then set working seq and pnat to MUTANT ones ...
    gale_set_seqarray(data_ptr, seqarray_mut, seqlen, num_seq);
    gale_set_pfold(data_ptr, pnat_mut);

    // iterate  design ...
    for ( i = 0; i < NNN; i++) {
        memcpy(seqarray_mut, seqarray, seqlen*num_seq*sizeof(unsigned char));
        fprintf(stderr, "iteration: %d \n", i);
        MutateSequences(seqarray_mut, seqlen, num_seq);

        gale_fold_compute(data_ptr, T);
        get_cost(seqarray_mut, seqlen, num_seq, pnat_mut, metab_cost, weight);
        
        ComparePnatSwapSeq( seqarray, seqarray_mut, pnat, pnat_mut, seqlen, num_seq );
        float avg_p = average(pnat, num_seq);
        if (avg_p>0.7) break;
        printf("%s --- iter %d  p %.4f ------\n", basename, i,avg_p);
    }
    
    if (i>NNN-2) break;  // break inner w-loop
    
    char aalist[]  = "CMFILVWYAGTSNQDEHRKPZ";
    int j;
    int aacount[20], totcount=0;
    for(i=0;i<20;i++) {
    aacount[i]=0; 
    for(j=0;j<num_seq*seqlen;j++) { if (seqarray[j]==i) { aacount[i]++; totcount++; } }
    }

    char fncost[100];
    sprintf(fncost, "%s-aacost.dat",basename);
    FILE *fpc = fopen(fncost,"w");
    for(i=0;i<20;i++) fprintf(fpc,"%c\t%.6f\n", aalist[i], metab_cost[i]);
    fclose(fpc);

    char fnc[100];
    sprintf(fnc,"%s-allfreq.dat",basename);
    FILE *fpp = fopen(fnc,"w");
    for(i=0;i<20;i++) fprintf(fpp,"%c\t%f\n",aalist[i],aacount[i]/(double)totcount);
    fclose(fpp);

    for(i=0;i<20;i++) {
    char fnout[100];
    sprintf(fnout, "L%d_w%.3f_freq%c.dat", seqlen, weight, aalist[i]);
    FILE *fout = fopen(fnout,"a");
    printf("%c %d  %d %f\n", aalist[i], aacount[i],totcount, (double)aacount[i]/(double)totcount);
    fprintf(fout, "%f\t%f\n",T,(double)aacount[i]/(double)totcount);
    fclose(fout);
    }
    
    char fnseq[100];
    sprintf(fnseq, "%s-seq.dat", basename);
    FILE *fps = fopen(fnseq,"w");
    for (int seq = 0; seq < num_seq; seq++) {
        for (int res = 0; res < seqlen; res++) {
            fprintf(fps, "%c", aalist[seqarray[seq*seqlen + res]]);
        }
        fprintf(fps, "\n");
    }
    fclose(fps);


    } // end w-loop


    } // end T-loop

    if(seqarray != NULL) { free(seqarray); }
    if(pnat != NULL) { free(pnat); }
    if(inat != NULL) { free(inat); }
    gale_close(&data_ptr);
    //
    return 0;
}

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

void GenerateSequences(unsigned char **sequences, int seqlen, int num_seq){
    //
    (*sequences) = (unsigned char *)malloc(num_seq*seqlen*sizeof(unsigned char));
    assert((*sequences) != NULL);
    for (int i = 0; i < num_seq*seqlen; i++) {
        (*sequences)[i] = (unsigned char)(rand() % 20); // random uint8 in range 0..19
    }
    // output just to check em out ...
    char AA[] = "CMFILVWYAGTSNQDEHRKP";
    FILE *fp = fopen("random_64_seq.d","w");
    for (int seq = 0; seq < num_seq; seq++) {
        for (int res = 0; res < seqlen; res++) {
            fprintf(fp, "%c", AA[(*sequences)[seq*seqlen + res]]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
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
