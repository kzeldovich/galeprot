#include "galelib.h"
#include <time.h>
// some stuff is already included in galelib.h ...
#define MAX(x,y) (((x)>(y))?(x):(y))
#define MIN(x,y) (((x)<(y))?(x):(y))
//
double diff(timespec start, timespec end);
inline unsigned char LetterToMJCode(char AAin);
int ReadFileWithSequences(const char *filename, unsigned char **ArrayOfSequences, int *MaxSequenceLength);
int ReadFileWithBindingPairs(const char *filename, int **bind_pairs);
//
int main(int argc, char **argv) {
    //
    timespec start, stop;
    int fold_wsize, bind_wsize;
    int num_conf, num_conf_bind, edge_len;
    //
    char *f_field_fname = argv[1];
    char *str_file_name = argv[2];
    if (sscanf(argv[3],"%d",&num_conf) != 1) { fprintf(stderr, "3rd arg is NumConf as int\n"); exit(1); }
    char *faces_b_fname = argv[4];
    if (sscanf(argv[5],"%d",&num_conf_bind) != 1) { fprintf(stderr, "5rd arg is NumConfBind as int\n"); exit(1); }
    if (sscanf(argv[6],"%d",&edge_len) != 1) { fprintf(stderr, "6rd arg is EdgeLen as int\n"); exit(1); }
    // in/out stuff ...
    char *seq_file_name = argv[7];
    if (sscanf(argv[8],"%d",&fold_wsize) != 1) { fprintf(stderr, "8th arg is FoldChunkSize as int\n"); exit(1); }
    char *bind_filename = argv[9];
    if (sscanf(argv[10],"%d",&bind_wsize) != 1) { fprintf(stderr, "10th arg is BindChunkSize as int\n"); exit(1); }
    char *fold_out_fname = argv[11];
    char *bind_out_fname = argv[12];
    // some checks ...
    if (num_conf != num_conf_bind) { fprintf(stderr, "NumConf must be equal to NumConfBind\n"); exit(1); }
    //
    // allocate some GALE struct ...
    GALE_Data *data_ptr;
    gale_init_null(&data_ptr,0);
    //
    // allocate sequences (load from somewhere ...) and binding pairs ...
    unsigned char *seqarray;
    int *bind_pairs;
    int seqlen, num_seq = ReadFileWithSequences(seq_file_name, &seqarray, &seqlen);
    int num_pairs = ReadFileWithBindingPairs(bind_filename, &bind_pairs);
    // conf_pairs: just allocating and setting to 0s, later pick up native folds ... 
    int *conf_pairs = (int *)malloc(2*num_pairs*sizeof(int)); assert(conf_pairs != NULL);
    memset(conf_pairs, 0, 2*num_pairs*sizeof(int));
    // allocate pnat & inat arrays, binding array to be allocated too ...
    float   *pnat = (float *)malloc(num_seq*sizeof(float));     assert(pnat != NULL);
    int     *inat = (int *)malloc(num_seq*sizeof(int));         assert(inat != NULL);
    float   *ebind = (float *)malloc(num_pairs*sizeof(float)); assert(ebind != NULL);
    float   *pbind = (float *)malloc(num_pairs*sizeof(float)); assert(pbind != NULL);
    int     *ibind = (int *)malloc(num_pairs*sizeof(int));     assert(ibind != NULL);
    //
    // load structural info, faces info & force field ...
    gale_load_forcefield(data_ptr, f_field_fname);
    gale_load_contactmap(data_ptr, str_file_name, num_conf);
    gale_load_bind_faces(data_ptr, faces_b_fname, num_conf, edge_len);
    //
    // define worksize (this must precede all other "set" type of functions)...
    gale_set_fold_worksize(data_ptr, fold_wsize);
    gale_set_bind_worksize(data_ptr, bind_wsize);
    //
    // connect seqarray to the device ...
    gale_set_seqarray(data_ptr, seqarray, seqlen, num_seq);
    gale_set_bind_pairs(data_ptr, bind_pairs, conf_pairs, num_pairs);
    gale_set_pfold(data_ptr, pnat);
    gale_set_ifold(data_ptr, inat);
    gale_set_pbind(data_ptr, pbind);
    gale_set_ibind(data_ptr, ibind);
    gale_set_ebind(data_ptr, ebind);
    // prepare & fold ...
    gale_fold_prepare(data_ptr);
    clock_gettime(CLOCK_REALTIME, &start);
    gale_fold_compute(data_ptr, 0.8);
    clock_gettime(CLOCK_REALTIME, &stop);
    //
    {
        double dt = diff(start, stop);
        printf("\nFolded %d seq in %.3f sec. %.2f kfps\n",num_seq, dt, ((float)num_seq)/dt/1000.0f);
    }
    //
    //
    // fill in conf_pairs with actual native folds ...
    for (int i = 0; i < 2*num_pairs; i++) {
        conf_pairs[i] = inat[bind_pairs[i]];
    }
    // prepare for bind & bind
    gale_fold_unprepare(data_ptr);
    gale_bind_prepare(data_ptr);
    clock_gettime(CLOCK_REALTIME, &start);
    gale_bind_compute(data_ptr, 0.8);
    clock_gettime(CLOCK_REALTIME, &stop);
    //
    {
        double dt = diff(start, stop);
        printf("\nBinded %d seq pairs in %.3f sec. %.2f kbps\n",num_pairs, dt, ((float)num_pairs)/dt/1000.0f);
    }

    //
    // some simple fold output ...
    {
        FILE *fp = fopen(fold_out_fname,"w");
        for (int i = 0; i < num_seq; i++) { fprintf(fp, "%d %.3f\n", inat[i], pnat[i]); }
        fclose(fp);
    }
    // some simple bind output ...
    {
        FILE *fp = fopen(bind_out_fname,"w"); 
        for (int i = 0; i < num_pairs; i++) { fprintf(fp, "%d %d %d %d %f %d %f\n", bind_pairs[2*i+0], bind_pairs[2*i+1], conf_pairs[2*i+0], conf_pairs[2*i+1], ebind[i], ibind[i], pbind[i]); } 
        fclose(fp);
    }
    //
    // freeing all allocated memory (user & API)...
    if(bind_pairs != NULL) { free(bind_pairs); }
    if(conf_pairs != NULL) { free(conf_pairs); }
    if(seqarray != NULL) { free(seqarray); }
    if(pnat != NULL) { free(pnat); }
    if(inat != NULL) { free(inat); }
    if(pbind != NULL) { free(pbind); }
    if(ebind != NULL) { free(ebind); }
    if(ibind != NULL) { free(ibind); }
    gale_close(&data_ptr);
    //
    return 0;
}





inline unsigned char LetterToMJCode(char AAin) {
    char AAandZ[] = "CMFILVWYAGTSNQDEHRKPZ";
    unsigned char aa_index = 0;
    // search a match from AAorder, 
    while ((AAin != AAandZ[aa_index])&&(aa_index < ALPHABET-1)) aa_index++;
    return aa_index;
}


int ReadFileWithSequences(const char *filename, unsigned char **ArrayOfSequences, int *MaxSequenceLength) {
    FILE *fp;
    int i,j;
    char SequenceBuffer[LINE_MAX];
    //
    assert(LINE_MAX > 1024);
    assert(INT_MAX > 1e+5);
    //
    int NumSeq = 0;
    int SeqLen, MaxSeqLen = 0, MinSeqLen = INT_MAX;
    //
    fp=fopen(filename,"r");
    if (fp == NULL) { fprintf(stderr, "no file with sequences\n"); exit(1); }
    //
    // read the file first to get NumSeq and MaxSeqLen ...
    while (fgets(SequenceBuffer, LINE_MAX, fp) != NULL) {
        NumSeq++;
        SeqLen = strlen(SequenceBuffer);
        // handle trailing newline symbol ...
        SeqLen = (SequenceBuffer[SeqLen-1] == '\n') ? (SeqLen-1) : SeqLen;
        MaxSeqLen = MAX(MaxSeqLen,SeqLen);
        MinSeqLen = MIN(MinSeqLen,SeqLen);
    }
    // once NumSeq and MaxSeqLen are calculated, allocate arrays and store stuff there ...
    printf("File with sequences is read: %d<=SequenceLength<=%d, NumberOfSequence=%d \n",MinSeqLen,MaxSeqLen,NumSeq);
    //
    (*ArrayOfSequences) = (unsigned char *)malloc(NumSeq*MaxSeqLen*sizeof(unsigned char));
    //
    // rewind to the file beginning ...
    rewind(fp);
    // as number of sequnces is known now ...
    for (i = 0; i < NumSeq; i++) {
        fgets(SequenceBuffer, LINE_MAX, fp);
        SeqLen = strlen(SequenceBuffer); 
        // handle trailing newline symbol ...
        SeqLen = (SequenceBuffer[SeqLen-1] == '\n') ? (SeqLen-1) : SeqLen;
        // fill in known sequence part & padd the remainder with 'Z' (MJcode=20) ...
        for ( j = 0; j < SeqLen; j++) {(*ArrayOfSequences)[i*MaxSeqLen + j] = LetterToMJCode(SequenceBuffer[j]); }
        for ( j = SeqLen; j < MaxSeqLen; j++) {(*ArrayOfSequences)[i*MaxSeqLen + j] = 20; }
    }
    // ArrayOfSequences is filled in.
    fclose(fp);
    *MaxSequenceLength = MaxSeqLen; // 'return' MaxSeqLen ...
    return NumSeq;//number of sequences read ...
}




int ReadFileWithBindingPairs(const char *filename, int **bind_pairs){
    FILE *fp = fopen(filename,"r");
    if (fp == NULL) { fprintf(stderr, "no file with binding pairs ...\n"); exit(1); }
    //
    int seq_first, seq_second;
    int num_pairs = 0;
    // loop over the requested number of pairs ...
    // file format - one pair per line: seq_id1 seq_id2 ...
    while (fscanf(fp, "%d %d", &seq_first, &seq_second) == 2){ num_pairs++; }
    printf("File with binding pairs is read: there are %d pairs to bind\n", num_pairs);
    // now as we know the number of binding pairs ...
    rewind(fp);
    // allocate pairs ...
    (*bind_pairs) = (int *)malloc(2*num_pairs*sizeof(int));
    // read the file all over again ...
    for (int i = 0; i < num_pairs; i++) {
        fscanf(fp, "%d %d", &seq_first, &seq_second);
        (*bind_pairs)[i*2+0] = seq_first;
        (*bind_pairs)[i*2+1] = seq_second;
    }
    fclose(fp);
    return num_pairs;
}



double diff(timespec start, timespec end) {
    timespec temp;
    if ((end.tv_nsec-start.tv_nsec)<0) {
        temp.tv_sec = end.tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
    } else {
        temp.tv_sec = end.tv_sec-start.tv_sec;
        temp.tv_nsec = end.tv_nsec-start.tv_nsec;
    }
    // return temp;
    return temp.tv_sec + temp.tv_nsec / 1000000000.;
}





























