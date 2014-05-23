 #include "galelib.h"
#include "kernels.h"


// #include cuda shmuda etc ...
#include <cuda_runtime.h>
#include <cublas_v2.h>


#define Safe_CUDA_Call(stmt)\
    {\
        cudaError_t err = stmt;\
        if (err != cudaSuccess) {\
            printf("CUDA ERROR, Failed to run: %s\n", #stmt);\
            printf("CUDA ERROR, Got CUDA error: '%s', line(%d)\n", cudaGetErrorString(err), __LINE__);\
            exit(1);\
        }\
    }\

#define Safe_CUBLAS_Call(stmt)\
    {\
        cublasStatus_t err = stmt;\
        if (err != CUBLAS_STATUS_SUCCESS) {\
            printf("ERROR, Failed to run something: %s\n", #stmt);\
            printf("ERROR, Got CUBLAS error with code %d, line(%d)\n", err, __LINE__);\
            exit(1);\
        }\
    }\




// assigns NULLs to every pointer in a GALE data struct ...
void gale_init(GALE_Data **gd, int deviceId){
    //
    // UNDER CONSTRUCTION ...
    int deviceCount, gpuDeviceCount = 0;
    struct cudaDeviceProp properties;
    cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
    if (cudaResultCode != cudaSuccess) { deviceCount = 0; }
    // machines with no GPUs can still report one emulation device
    for (int device = 0; device < deviceCount; device++) {
        Safe_CUDA_Call(cudaGetDeviceProperties(&properties, device));
        // 9999 means emulation only
        if (properties.major != 9999) { gpuDeviceCount++; }
    }
    if (gpuDeviceCount < 1) { fprintf(stderr, "\n\nCUDA enabled GPU is not found!\n\n"); exit(1); }
    printf("%d GPU CUDA device(s) found\n", gpuDeviceCount);
    //
    // now setting device with a given id ...
    Safe_CUDA_Call(cudaSetDevice(deviceId));
    //
    //
    size_t free_device_mem, total_device_mem;
    Safe_CUDA_Call(cudaMemGetInfo(&free_device_mem, &total_device_mem));
    printf("CUDA launched with device memory: %zu bytes are free on init stage (total %zu)\n", free_device_mem, total_device_mem);
    // UNDER CONSTRUCTION ...
    //
    //
    // allocating struct itself first ...
    (*gd) = (GALE_Data *)malloc(sizeof(GALE_Data));
    //
    //
    (*gd)->outlist_fold = 0x00;
    (*gd)->outlist_bind = 0x00;
    (*gd)->report_level = GALE_REPORTS_DEFAULT;
    //
    //
    (*gd)->contactmap_host = NULL;
    (*gd)->forcefield_host = NULL;
    (*gd)->seqarray_host = NULL;
    (*gd)->pfold_host = NULL; 
    (*gd)->gfold_host = NULL; 
    (*gd)->ifold_host = NULL;
    (*gd)->efold_host = NULL; 
    (*gd)->sfold_host = NULL; 
    // binding related ...
    (*gd)->bind_seq_pairs_host = NULL;
    (*gd)->bind_conf_pairs_host = NULL;
    (*gd)->bind_faces_host = NULL;
    (*gd)->pbind_host = NULL;
    (*gd)->gbind_host = NULL;
    (*gd)->ibind_host = NULL;
    (*gd)->ebind_host = NULL;
    (*gd)->sbind_host = NULL;
    //
    //DEVICE
    // general & folding related ...
    (*gd)->seqarray_device = NULL;
    (*gd)->idxlut_device = NULL;
    (*gd)->contactmap_matrix_device = NULL;
    (*gd)->forcefield_device = NULL;
    (*gd)->seq_evector_device = NULL;
    (*gd)->pfold_device = NULL; 
    (*gd)->gfold_device = NULL; 
    (*gd)->ifold_device = NULL;
    (*gd)->efold_device = NULL; 
    (*gd)->sfold_device = NULL;
    // binding related ...
    (*gd)->bind_seq_pairs_device = NULL;
    (*gd)->bind_conf_pairs_device = NULL;
    (*gd)->bind_faces_device = NULL;
    // (*gd)->seqseq_evectors_device = NULL;
    // size_t binding_residues_size = num_pairs*C144*edge_squared*sizeof(int);
    (*gd)->bind_residues_device = NULL;
    (*gd)->pbind_device = NULL;
    (*gd)->gbind_device = NULL;
    (*gd)->ibind_device = NULL;
    (*gd)->ebind_device = NULL;
    (*gd)->sbind_device = NULL;
    //
    //
    // VARIABLES ...
    (*gd)->max_seqlen = 0;
    (*gd)->max_numcontacts = 0;
    (*gd)->existing_contacts = 0;
    (*gd)->num_seq = 0;
    (*gd)->num_conf = 0;
    // binding only ...
    (*gd)->num_conf_bind = 0; // this should be equal to num_conf ... but who knows ...
    (*gd)->edge_len = 0;
    // edge^2 ...
    (*gd)->face_size = 0;
    (*gd)->num_pairs = 0;
    // chunks ...
    (*gd)->fold_chunk_size = 0;
    (*gd)->bind_chunk_size = 0;
    //
    return;
}
// loading force field, along with allocating it on the device ...
void gale_load_forcefield(GALE_Data *gd, const char *fname){
    //
    int i;
    // connect AA alphabet with MJ 0-19 codes ...
    char aminoacids[] = "CMFILVWYAGTSNQDEHRKP";
    // create lookup ...
    unsigned char aa_lut[256];
    for (i = 0; i < ALPHABET-1; i++) { aa_lut[aminoacids[i]] = i; }
    // now aa_lut['C']==0 and aa_lut['P']==19 ...
    char aa1, aa2; 
    float e;
    size_t forcefield_size = ALPHABET*ALPHABET*sizeof(float);
    // allocating host and device forcefiled arrays ...
    gd->forcefield_host = (float *)malloc(forcefield_size);
    assert(gd->forcefield_host != NULL);
    // all values are initially set to zeroes ...
    memset(gd->forcefield_host, 0, forcefield_size);
    // 
    FILE *fp = fopen(fname,"r"); 
    if (fp == NULL) { fprintf(stderr, "cannot read %s\n",fname); exit(1); } 
    //
    i = 0;
    // fscanf and %c behave strangely with the \n char: now it works 
    while( fscanf(fp, " %c %c %f\n", &aa1, &aa2, &e) == 3){
        gd->forcefield_host[ALPHABET*aa_lut[aa1] + aa_lut[aa2]] = e;
        i++;
    }
    if (i != (ALPHABET-1)*(ALPHABET-1)) {fprintf(stderr, "400 entries expected in %s\n",fname); exit(1);};
    fclose(fp);
    // some reports ...
    if (gd->report_level >= GALE_REPORTS_DEFAULT) {printf("loaded %s as Force Field (non-normalized)\n", fname);}
    // device allocation and copy ...
    Safe_CUDA_Call(cudaMalloc((void**)&gd->forcefield_device, forcefield_size ));
    Safe_CUDA_Call(cudaMemcpy(gd->forcefield_device, gd->forcefield_host, forcefield_size, cudaMemcpyHostToDevice));
    // forcefield copied to device memory ...
}
// auto load mode so far ...
void gale_load_contactmap(GALE_Data *gd, const char *fname, int num_conf){
    //
    int i,j,k;
    int fscanf_return = 0;
    int c1, c2, tmp;
    int StrIndexNext, StrIndexPrev;
    int NumOfConts;
    int LocalMaxNumOfConts, LocalMinNumOfConts;
    int StructCounter;
    int max_seqlen_conf_estimate = 0;
    // open file if it's possible ...
    FILE *fp = fopen(fname,"r");
    if (fp == NULL) { fprintf(stderr, "cannot read %s\n",fname); exit(1); }
    //
    // calculate MaxNumOfConts first (read the whole file)...
    StrIndexPrev = 0; // this assumption must be documented ...
    LocalMaxNumOfConts = NumOfConts = 0;
    LocalMinNumOfConts = INT_MAX;
    StructCounter = 0;
    //
    // loop over the requested number of structures ...
    while ( fscanf(fp, "%d %d %d", &StrIndexNext, &c1, &c2) == 3 ){
        //
        if (StrIndexPrev == StrIndexNext) {
            NumOfConts++;
        } else {
            LocalMaxNumOfConts = max( LocalMaxNumOfConts, NumOfConts );
            LocalMinNumOfConts = min( LocalMinNumOfConts, NumOfConts );
            NumOfConts = 1; // 1 is correct here
            StructCounter++;
            StrIndexPrev = StrIndexNext;
        }
    }
    // finish after EOF ...
    LocalMaxNumOfConts = max( LocalMaxNumOfConts, NumOfConts );
    LocalMinNumOfConts = min( LocalMinNumOfConts, NumOfConts );
    StructCounter++;
    // allocate ContactMap right in here and fill in the gdata params ...
    gd->max_numcontacts = LocalMaxNumOfConts;
    gd->num_conf = StructCounter;
    // small check ... TO BE UPDATED LATER ...
    if (gd->num_conf != num_conf) {
        fprintf(stderr, "Contact map data from %s does not comply with the format.\n", fname);
        exit(1);
    }
    // small check ... TO BE UPDATED LATER ...
    gd->contactmap_host = (unsigned char *)malloc(2 * gd->max_numcontacts * gd->num_conf *sizeof(unsigned char));
    assert(gd->contactmap_host != NULL);
    unsigned char * MaxContBufferA = (unsigned char *)malloc(gd->max_numcontacts*sizeof(unsigned char));
    assert(MaxContBufferA != NULL);
    unsigned char * MaxContBufferB = (unsigned char *)malloc(gd->max_numcontacts*sizeof(unsigned char));
    assert(MaxContBufferB != NULL);
    //
    if (gd->report_level >= GALE_REPORTS_DEFAULT) {printf(  "loaded %s as contact map\n", fname);}
    if (gd->report_level >= GALE_REPORTS_VERBOSE) {printf(  "  number of contacts range: (%d,%d)\n"
                                                            "  number of conformations: %d\n",
                                                            LocalMinNumOfConts,LocalMaxNumOfConts,gd->num_conf);}
    //
    // now we know MaxNumOfStruct ...
    // rewind to the file beginning ...
    rewind(fp);
    // read the file all over again ...
    // this time filling ContactMap with real data ...
    fscanf(fp, "%d %d %d", &StrIndexPrev, &c1, &c2);
    for (i = 0; i < gd->num_conf; i++) {
        //
        j = 0;
        do { // read all conts from a given structure ... 
            MaxContBufferA[j] = c1;
            MaxContBufferB[j] = c2;
            // max_seqlen estimate from structural data is simply the biggest residue index ever occured ...
            // BEWARE ci -are numbered starting with 0! N-1 should yield N of length ...
            max_seqlen_conf_estimate = max(max_seqlen_conf_estimate,max(c1+1,c2+1));
            j++;
            fscanf_return = fscanf(fp, "%d %d %d", &StrIndexNext, &c1, &c2);
        } while ( (StrIndexPrev == StrIndexNext) && (fscanf_return == 3) );
        //
        StrIndexPrev = StrIndexNext;
        // copy from buffer into big padded array
        for (k = 0; k < gd->max_numcontacts; k++) {
            //
            // if NumOfCont(j) exceeded -> fillin last element that makes sense ...
            gd->contactmap_host[i*2*gd->max_numcontacts + 2*k + 0] = MaxContBufferA[min(k,j-1)];
            gd->contactmap_host[i*2*gd->max_numcontacts + 2*k + 1] = MaxContBufferB[min(k,j-1)];
            //
        }
    }// COntactMap is filled in 100% now ...
    //
    fclose(fp);
    free(MaxContBufferA);
    free(MaxContBufferB);
    ////////////////////////////////////////////////////////////////////////
    // NEXT: GENERATING CONTACT MAP SUITABLE FOR THE DEVICE ...
    ////////////////////////////////////////////////////////////////////////
    //
    int indexA, indexB, tmp_index;
    int seqlen_estimate_square = max_seqlen_conf_estimate*max_seqlen_conf_estimate;
    //
    int ExistingContCounter=0;
    //
    int *layover_contactmap_local = (int *)malloc(seqlen_estimate_square*sizeof(int));
    // zero out this thing ...
    memset(layover_contactmap_local, 0, seqlen_estimate_square*sizeof(int));
    // overlay all contact maps from all structures to get used-contacts ...
    for(i = 0; i < gd->num_conf; i++) { 
        for(j=0; j < gd->max_numcontacts; j++) {
            indexA = gd->contactmap_host[2*(i*gd->max_numcontacts + j) + 0];
            indexB = gd->contactmap_host[2*(i*gd->max_numcontacts + j) + 1];
            // make sure A < B ...
            tmp_index = min(indexA,indexB);
            indexB = max(indexA,indexB);
            indexA = tmp_index;
            // matrix is ALWAYS symmetric, fill the upper triangle only ...
            layover_contactmap_local[indexA*max_seqlen_conf_estimate + indexB] = 1;
        }
    }
    // count number of contact that are actually present in all strucural ContMaps
    for(i = 0; i < max_seqlen_conf_estimate; i++) {
        // use upper triangle only ...
        for(j = i; j < max_seqlen_conf_estimate; j++) {
            ExistingContCounter += layover_contactmap_local[i*max_seqlen_conf_estimate + j];
        }
    }
    // helping info and statistics about loaded contact maps ...
    gd->existing_contacts = ExistingContCounter;
    int abs_max_contacts = max_seqlen_conf_estimate*(max_seqlen_conf_estimate-1)/2;
    float contacts_sparsity = (float)(gd->existing_contacts)/(float)abs_max_contacts;
    assert( gd->existing_contacts <= abs_max_contacts );
    // extra verbose contacts stat report here ...
    if (gd->report_level >= GALE_REPORTS_VERBOSE) {
        printf( "Given contacts map statistics ...\n"
                "  estimated max sequence length: %d\n"
                "  theoretical max number of contacts: %d\n"
                "  observed num of contacts: %d (contacts sparsity: %.3f)\n",
                max_seqlen_conf_estimate,abs_max_contacts,gd->existing_contacts,contacts_sparsity);}
    // allocations local and for our struct ...
    size_t idxlut_size = 2* gd->existing_contacts *sizeof(unsigned char);
    size_t contactmap_matrix_size = 2*gd->existing_contacts*gd->num_conf*sizeof(float);
    unsigned char *idxlut_host_local = (unsigned char *)malloc(idxlut_size);
    float *contactmap_matrix_host_local = (float *)malloc(contactmap_matrix_size);
    // fill in the contactmap_matrix_host_local with zeroes ...
    memset(contactmap_matrix_host_local, 0, contactmap_matrix_size);
    //
    // fill in the idx_lookup table
    k = 0;
    for(i = 0; i < max_seqlen_conf_estimate; i++) {
        for(j = i; j < max_seqlen_conf_estimate; j++) {
            // store meaningful pairs only ...
            if (layover_contactmap_local[i*max_seqlen_conf_estimate + j]) {idxlut_host_local[k]=i; idxlut_host_local[k+1] = j; k += 2;}
        }
    }
    //
    // fill in the index matrix with zeroes and ones ...
    for(i = 0; i < gd->num_conf; i++) {
        for(j = 0; j < 2*gd->max_numcontacts; j+=2) {
            c1 = gd->contactmap_host[2*i*gd->max_numcontacts + j + 0];
            c2 = gd->contactmap_host[2*i*gd->max_numcontacts + j + 1];
            // make sure c1<c2 - upper triangle of the contact map ...
            tmp = min(c1,c2); c2 = max(c1,c2); c1 = tmp; // now c1<c2 for sure!
            // rather slow exhaustive search here for index in idx_lut ...
            for(k = 0; k < ExistingContCounter; k++) {
                if ((idxlut_host_local[2*k]==c1)&&(idxlut_host_local[2*k+1]==c2)){
                    // we need it to be float for further GPU processing ...
                    contactmap_matrix_host_local[k*gd->num_conf + i] = 1.0f;
                }
            }
        }
    }
    // device allocation and copy ...
    // index look up table ...
    Safe_CUDA_Call(cudaMalloc((void**)&gd->idxlut_device, idxlut_size ));
    Safe_CUDA_Call(cudaMemcpy(gd->idxlut_device, idxlut_host_local, idxlut_size, cudaMemcpyHostToDevice));
    // contactmap_matrix_device - contact map in a form of zero/ones matrix ...
    Safe_CUDA_Call(cudaMalloc((void**)&gd->contactmap_matrix_device, contactmap_matrix_size ));
    Safe_CUDA_Call(cudaMemcpy(gd->contactmap_matrix_device, contactmap_matrix_host_local, contactmap_matrix_size, cudaMemcpyHostToDevice));
    //
    if (gd->report_level >= GALE_REPORTS_DEFAULT) {printf("contact map successfully converted for device use.\n"); }
    if (gd->report_level >= GALE_REPORTS_VERBOSE) {printf("  fromated as a sparse zero-one matrix, which is to device.\n");}
    free(layover_contactmap_local);
    free(contactmap_matrix_host_local);
    //
    return;
    //
    //
}
// auto load mode so far, handles cubic proteins only ...
void gale_load_bind_faces(GALE_Data *gd, const char *fname, int num_conf, int edge_length){
    // open file if it's possible ...
    FILE *fp = fopen(fname,"r");
    if (fp == NULL) { fprintf(stderr, "no file with faces: %s\n",fname); exit(1); }
    //
    char face_in_orientation[LINE_MAX];
    char *ptr_to_read_from;
    int tmp_var;
    int chars_read;
    int numbers_in_line = 0;
    int numbers_in_line_prev = 0;
    int lines = 0;
    //
    while (fgets(face_in_orientation, LINE_MAX, fp) != NULL) {
        //
        ptr_to_read_from = face_in_orientation;
        numbers_in_line = 0;
        while (sscanf(ptr_to_read_from,"%d%n",&tmp_var,&chars_read)==1){
            numbers_in_line++;
            ptr_to_read_from += chars_read;
        }
        // if it's a first line read, remember # of numbers in line ...
        if (lines == 0) { numbers_in_line_prev = numbers_in_line; } else {
            // # of nums on every line must stay the same ...
            if (numbers_in_line_prev != numbers_in_line) {
                fprintf(stderr, "bad data in faces file %s, on the line %d\n",fname,(lines+1));
                exit(1);
            }
        }
        //
        lines++;
        //
    }
    // now we know # of lines, and # of numbers in every line - which is the same!
    // according to the data format we're expecting we can deduce seq_len (edge_len)...
    int edge_len;
    int conformations = lines/(FACES*ORIENTATIONS); // 6 faces and 4 orientations ...
    edge_len = (int)sqrt(numbers_in_line - 3);
    // if something doenst match our expectations ...
    if ((edge_len*edge_len+3 != numbers_in_line)||(conformations*FACES*ORIENTATIONS != lines)) {
        fprintf(stderr, "In faces file %s, each line must look like: conf face orient monomer1 ... monomerM\n",fname);
        fprintf(stderr, "here conf - struct index 0..N; face: 0..5; orient 0..3. Number of monomers M is a perfect square \n");
        fprintf(stderr, "because there are EDGE*EDGE monomers on each face!\n");
        fprintf(stderr, "Make sure you're feeding correct data for this program first. EXIT\n");
        exit(1);
    }
    // allocations ...
    gd->edge_len = edge_len;
    gd->face_size = edge_len * edge_len;
    gd->num_conf_bind = conformations;
    // small check ... TO BE UPDATED LATER ...
    if ((gd->edge_len != edge_length)||(gd->num_conf_bind != num_conf)) {
        fprintf(stderr, "Binding faces data from %s does not comply with the specification.\n", fname);
        exit(1);
    }
    // small check ... TO BE UPDATED LATER ...
    size_t bind_faces_size = gd->num_conf_bind * FACES*ORIENTATIONS* gd->face_size *sizeof(unsigned char);
    gd->bind_faces_host = (unsigned char *)malloc(bind_faces_size);
    assert(gd->bind_faces_host != NULL);
    Safe_CUDA_Call(cudaMalloc((void**)&gd->bind_faces_device, bind_faces_size ));
    //
    // is everything is fine: rewind and read data in the proper arrays ...
    rewind(fp);
    //
    int i, j;
    int conf, face, orient;
    int residue_index;
    for (i = 0; i < gd->num_conf_bind*FACES*ORIENTATIONS; i++) {
        // for all faces in all orientations fill in gd->bind_faces_host ...
        fgets(face_in_orientation, LINE_MAX, fp);
        // get conf,face,orient
        sscanf(face_in_orientation,"%d %d %d%n",&conf,&face,&orient,&chars_read);
        ptr_to_read_from = face_in_orientation + chars_read;
        for (j = 0; j < gd->face_size; j++){
            //
            sscanf(ptr_to_read_from,"%d%n",&residue_index,&chars_read);
            gd->bind_faces_host[(conf*FACES*ORIENTATIONS + face*ORIENTATIONS + orient)*gd->face_size + j] = residue_index;
            ptr_to_read_from += chars_read;
        }

    }
    fprintf(stderr,"File with faces is read: EdgeLength=%d, NumberOfConfsBind=%d\n",gd->edge_len,gd->num_conf_bind);
    fclose(fp);
    // copy faces to device mem ...
    Safe_CUDA_Call(cudaMemcpy(gd->bind_faces_device, gd->bind_faces_host, bind_faces_size, cudaMemcpyHostToDevice));
    return;
}


// SET FUNCTIONS ...

// SERVICE SET FUNCTIONS ...
// size of the chunks to be processed as 1 grid on a device ...
void gale_set_fold_worksize(GALE_Data *gd, int fold_worksize){
    gd->fold_chunk_size = fold_worksize;
    if (gd->report_level >= GALE_REPORTS_VERBOSE) {
        printf("Number of seq to fold on GPU at the same time is set: %d\n", gd->fold_chunk_size);
    }
    //
    return;
}
// size of the chunks to be processed as 1 grid on a device ...
void gale_set_bind_worksize(GALE_Data *gd, int bind_worksize){
    gd->bind_chunk_size = bind_worksize;
    if (gd->report_level >= GALE_REPORTS_VERBOSE) {
        printf("Number of seq pairs to bind on GPU at the same time is set: %d\n", gd->bind_chunk_size);
    }
    //
    return;
}
// set report level: quiet, verbose, debug etc ...
void gale_set_report_level(GALE_Data *gd, int report_level){
    gd->report_level = report_level;
}
// set otuputting arrays of compute_fold ...
void gale_set_output_fold(GALE_Data *gd, int fold_output_list){
    gd->outlist_fold |= fold_output_list;
}
// set otuputting arrays of compute_fold ...
void gale_set_output_bind(GALE_Data *gd, int bind_output_list){
    gd->outlist_bind |= bind_output_list;
}
// SERVICE SET FUNCTIONS ...



// set seqarray for the first time, initializing seqlen and num sequences ...
void gale_set_seqarray(GALE_Data *gd, unsigned char *seqarray, int seqlen, int num_seq){
    // struct variables ...
    gd->max_seqlen = seqlen;
    gd->num_seq = num_seq;
    //
    gd->seqarray_host = seqarray;
    return;
}
// set binding pairs ...
void gale_set_bind_pairs(GALE_Data *gd, int *seq_pairs, int *conf_pairs, int num_pairs){
    gd->num_pairs = num_pairs;
    //
    gd->bind_seq_pairs_host = seq_pairs;
    gd->bind_conf_pairs_host = conf_pairs;
    return;
}
// FOLD RESULTS SET ...
// set (assign) host pnat array ...
void gale_set_pfold(GALE_Data *gd, float *pfold_array){
    gd->pfold_host = pfold_array;
    gd->outlist_fold |= GALE_RETURN_PNAT_FOLD;
    return;
}
// set (assign) host deltaG array ...
void gale_set_gfold(GALE_Data *gd, float *gfold_array){
    gd->gfold_host = gfold_array;
    gd->outlist_fold |= GALE_RETURN_DG_FOLD;
    return;
}
// set (assign) host native indexes array ...
void gale_set_ifold(GALE_Data *gd, int *ifold_array){
    gd->ifold_host = ifold_array;
    gd->outlist_fold |= GALE_RETURN_INDEX_FOLD;
    return;
}
// set (assign) host Emin array ...
void gale_set_efold(GALE_Data *gd, float *efold_array){
    gd->efold_host = efold_array;
    gd->outlist_fold |= GALE_RETURN_EMIN_FOLD;
    return;
}
// set (assign) host spectrum array ...
void gale_set_sfold(GALE_Data *gd, float *sfold_array){
    gd->sfold_host = sfold_array;
    gd->outlist_fold |= GALE_RETURN_SPECTRUM_FOLD;
    return;
}
// BIND RESULTS SET ...
// set (assign) host pnat bind array ...
void gale_set_pbind(GALE_Data *gd, float *pbind_array){
    gd->pbind_host = pbind_array;
    gd->outlist_bind |= GALE_RETURN_PNAT_BIND;
    return;
}
// set (assign) host bind free energy array ...
void gale_set_gbind(GALE_Data *gd, float *gbind_array){
    gd->gbind_host = gbind_array;
    gd->outlist_bind |= GALE_RETURN_DG_BIND;
    return;
}
// set (assign) host native indexes binding array ...
void gale_set_ibind(GALE_Data *gd, int *ibind_array){
    gd->ibind_host = ibind_array;
    gd->outlist_bind |= GALE_RETURN_INDEX_BIND;
    return;
}
// set (assign) host native emin binding array ...
void gale_set_ebind(GALE_Data *gd, float *ebind_array){
    gd->ebind_host = ebind_array;
    gd->outlist_bind |= GALE_RETURN_EMIN_BIND;
    return;
}
// set (assign) host 144 binding spectrum array ...
void gale_set_sbind(GALE_Data *gd, float *sbind_array){
    gd->sbind_host = sbind_array;
    gd->outlist_bind |= GALE_RETURN_SPECTRUM_BIND;
    return;
}

// prepare functions to perform device allocations ...
void gale_fold_prepare(GALE_Data *gd){
    //
    // clean up folding allocations (do we need to do this in a separate function????) 
    if (gd->seqarray_device != NULL) { Safe_CUDA_Call(cudaFree(gd->seqarray_device)); }
    if (gd->seq_evector_device != NULL) { Safe_CUDA_Call(cudaFree(gd->seq_evector_device)); }
    if (gd->pfold_device != NULL) { Safe_CUDA_Call(cudaFree(gd->pfold_device)); }
    if (gd->gfold_device != NULL) { Safe_CUDA_Call(cudaFree(gd->gfold_device)); }
    if (gd->ifold_device != NULL) { Safe_CUDA_Call(cudaFree(gd->ifold_device)); }
    if (gd->efold_device != NULL) { Safe_CUDA_Call(cudaFree(gd->efold_device)); }
    if (gd->sfold_device != NULL) { Safe_CUDA_Call(cudaFree(gd->sfold_device)); }
    //
    //
    Safe_CUDA_Call(cudaMalloc((void**)&gd->seqarray_device, gd->num_seq*gd->max_seqlen*sizeof(unsigned char) ));
    //
    // knowing chunks size ...
    // allocate device arrays that rely on the chunk_size here ...
    // work or chunk size is to replace num_seq or num_pairs to be processed per batch ...
    Safe_CUDA_Call(cudaMalloc((void **)&gd->seq_evector_device, gd->fold_chunk_size*gd->existing_contacts*sizeof(float) ));
    Safe_CUDA_Call(cudaMalloc((void **)&gd->pfold_device, gd->fold_chunk_size*sizeof(float) ));
    Safe_CUDA_Call(cudaMalloc((void **)&gd->gfold_device, gd->fold_chunk_size*sizeof(float) ));
    Safe_CUDA_Call(cudaMalloc((void **)&gd->ifold_device, gd->fold_chunk_size*sizeof(int)));
    Safe_CUDA_Call(cudaMalloc((void **)&gd->efold_device, gd->fold_chunk_size*sizeof(float) ));
    Safe_CUDA_Call(cudaMalloc((void **)&gd->sfold_device, gd->fold_chunk_size*gd->num_conf*sizeof(float) ));
    //
    //
    size_t free_device_mem, total_device_mem;
    Safe_CUDA_Call(cudaMemGetInfo(&free_device_mem, &total_device_mem));
    float device_mem_utilization = 1.0f - (float)free_device_mem/(float)total_device_mem;
    if (device_mem_utilization > 0.8f) {
        fprintf(stderr, "\n\nBEWARE: GPU memory utilization > 80%%! Reduce the FOLD chunk size.\n\n");
    }
    if (gd->report_level >= GALE_REPORTS_VERBOSE) {
        printf("GPU memory utilization at fold prepare %.1f%%:\n", device_mem_utilization*100.0f);
        printf("total: %zu bytes; free: %zu  bytes\n", total_device_mem, free_device_mem);
    }
    //
    //
    return;
}
void gale_bind_prepare(GALE_Data *gd){
    //
    // clean up binding allocations (do we need to do this in a separate function????) ...
    if (gd->bind_seq_pairs_device != NULL) { Safe_CUDA_Call(cudaFree(gd->bind_seq_pairs_device)); }
    if (gd->bind_conf_pairs_device != NULL) { Safe_CUDA_Call(cudaFree(gd->bind_conf_pairs_device)); }
    // if (gd->seqseq_evectors_device != NULL) { Safe_CUDA_Call(cudaFree(gd->seqseq_evectors_device)); }
    if (gd->bind_residues_device != NULL) { Safe_CUDA_Call(cudaFree(gd->bind_residues_device)); }    
    if (gd->pbind_device != NULL) { Safe_CUDA_Call(cudaFree(gd->pbind_device)); }
    if (gd->gbind_device != NULL) { Safe_CUDA_Call(cudaFree(gd->gbind_device)); }
    if (gd->ibind_device != NULL) { Safe_CUDA_Call(cudaFree(gd->ibind_device)); }
    if (gd->ebind_device != NULL) { Safe_CUDA_Call(cudaFree(gd->ebind_device)); }
    if (gd->sbind_device != NULL) { Safe_CUDA_Call(cudaFree(gd->sbind_device)); }
    // do we need to do this in a separate function????
    //
    Safe_CUDA_Call(cudaMalloc((void**)&gd->bind_seq_pairs_device, 2*gd->num_pairs*sizeof(int) ));
    Safe_CUDA_Call(cudaMalloc((void**)&gd->bind_conf_pairs_device, 2*gd->num_pairs*sizeof(int) ));
    //
    // allocate device arrays that rely on the chunk_size here ...
    // work or chunk size is to replace num_seq or num_pairs to be processed per batch ...
    // Safe_CUDA_Call(cudaMalloc((void **)&gd->seqseq_evectors_device, gd->bind_chunk_size*gd->max_seqlen*gd->max_seqlen*sizeof(float)));
    Safe_CUDA_Call(cudaMalloc((void **)&gd->bind_residues_device, gd->bind_chunk_size*C144*gd->face_size*sizeof(int)));
    Safe_CUDA_Call(cudaMalloc((void **)&gd->pbind_device, gd->bind_chunk_size*sizeof(float)));
    Safe_CUDA_Call(cudaMalloc((void **)&gd->gbind_device, gd->bind_chunk_size*sizeof(float)));
    Safe_CUDA_Call(cudaMalloc((void **)&gd->ibind_device, gd->bind_chunk_size*sizeof(int)));
    Safe_CUDA_Call(cudaMalloc((void **)&gd->ebind_device, gd->bind_chunk_size*sizeof(float)));
    Safe_CUDA_Call(cudaMalloc((void **)&gd->sbind_device, gd->bind_chunk_size*C144*sizeof(float)));
    //
    size_t free_device_mem, total_device_mem;
    Safe_CUDA_Call(cudaMemGetInfo(&free_device_mem, &total_device_mem));
    if ((float)free_device_mem/(float)total_device_mem < 0.2f) {
        fprintf(stderr, "\n\nBEWARE: GPU memory utilization > 80%%! Reduce the BIND chunk size.\n\n");
    }
    //
    return;
}


void gale_fold_unprepare(GALE_Data *gd){
    // what to do with seqarray ??????????????????????
    // if (gd->seqarray_device != NULL) { Safe_CUDA_Call(cudaFree(gd->seqarray_device)); }
    // if (gd->seq_evector_device != NULL) { Safe_CUDA_Call(cudaFree(gd->seq_evector_device)); }
    if (gd->pfold_device != NULL) { Safe_CUDA_Call(cudaFree(gd->pfold_device)); gd->pfold_device = NULL;}
    if (gd->gfold_device != NULL) { Safe_CUDA_Call(cudaFree(gd->gfold_device)); gd->gfold_device = NULL;}
    if (gd->ifold_device != NULL) { Safe_CUDA_Call(cudaFree(gd->ifold_device)); gd->ifold_device = NULL;}
    if (gd->efold_device != NULL) { Safe_CUDA_Call(cudaFree(gd->efold_device)); gd->efold_device = NULL;}
    if (gd->sfold_device != NULL) { Safe_CUDA_Call(cudaFree(gd->sfold_device)); gd->sfold_device = NULL;}
}
void gale_bind_unprepare(GALE_Data *gd){
    // what to do with seqarray ??????????????????????
    // clean up binding allocations (do we need to do this in a separate function????) ...
    if (gd->bind_seq_pairs_device != NULL) { Safe_CUDA_Call(cudaFree(gd->bind_seq_pairs_device)); gd->bind_seq_pairs_device = NULL; }
    if (gd->bind_conf_pairs_device != NULL) { Safe_CUDA_Call(cudaFree(gd->bind_conf_pairs_device)); gd->bind_conf_pairs_device = NULL; }
    // if (gd->seqseq_evectors_device != NULL) { Safe_CUDA_Call(cudaFree(gd->seqseq_evectors_device)); gd->seqseq_evectors_device = NULL; }
    if (gd->bind_residues_device != NULL) { Safe_CUDA_Call(cudaFree(gd->bind_residues_device)); gd->bind_residues_device = NULL; }
    if (gd->pbind_device != NULL) { Safe_CUDA_Call(cudaFree(gd->pbind_device)); gd->pbind_device = NULL; }
    if (gd->gbind_device != NULL) { Safe_CUDA_Call(cudaFree(gd->gbind_device)); gd->gbind_device = NULL; }
    if (gd->ibind_device != NULL) { Safe_CUDA_Call(cudaFree(gd->ibind_device)); gd->ibind_device = NULL; }
    if (gd->ebind_device != NULL) { Safe_CUDA_Call(cudaFree(gd->ebind_device)); gd->ebind_device = NULL; }
    if (gd->sbind_device != NULL) { Safe_CUDA_Call(cudaFree(gd->sbind_device)); gd->sbind_device = NULL; }
}



// fold ...
void gale_fold_compute(GALE_Data *gd, float fold_temp){
    // folding in chuncks ...
    // allocations for cuBLAS ...
    cublasHandle_t handle;
    Safe_CUBLAS_Call(cublasCreate(&handle));
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    //
    float fold_exp_coeff = -1.0f/fold_temp;
    //
    dim3 grid(1,1,1);
    dim3 block2D(16,16,1);
    // we'd need array of mutexes for new min/max reduction function ...
    int *mutex_array;
    Safe_CUDA_Call(cudaMalloc((void**)&mutex_array, gd->fold_chunk_size*sizeof(int)));
    Safe_CUDA_Call(cudaMemset(mutex_array, 0, gd->fold_chunk_size*sizeof(int)));
    //
    // sequences to fold are reffered by gd->seqarray_host ...
    // copy them from host to device ...
    Safe_CUDA_Call(cudaMemcpy(gd->seqarray_device, gd->seqarray_host, gd->num_seq*gd->max_seqlen*sizeof(unsigned char), cudaMemcpyHostToDevice));
    //
    int i, fold_num_chunks = SPLIT(gd->num_seq,gd->fold_chunk_size);
    if (gd->report_level >= GALE_REPORTS_VERBOSE) {
        printf("Number of chunks is calculated in fold function:\n%d sequences to fold in %d chunks.\n",gd->num_seq,fold_num_chunks);
    }
    // MAIN LOOP TO PROCESS DATA IN CHUNKS IS HERE ...
    for (i = 0; i < fold_num_chunks; i++) {
        // update initial state ...
        grid.x = 1; grid.y = 1; grid.z = 1; 
        block2D.x = 16; block2D.y = 16; block2D.z = 1; 
        // all of the arrays are
        unsigned char *seqarray_chunk = gd->seqarray_device + i*gd->fold_chunk_size*gd->max_seqlen;
        int iter_chunk_size = (i < fold_num_chunks-1) ? (gd->fold_chunk_size) : (gd->num_seq - i*gd->fold_chunk_size);
        //
        // form seq_evectors using idxlut, given sequences, and forcefileld .... 
        grid.x = SPLIT(gd->existing_contacts,block2D.x); grid.y = SPLIT(iter_chunk_size,block2D.y);
        kernel_convert_seq_lut_v2<<<grid,block2D>>>(    seqarray_chunk, gd->seq_evector_device,
                                                        gd->forcefield_device, gd->idxlut_device,
                                                        gd->max_seqlen, gd->existing_contacts,
                                                        iter_chunk_size);
        // calculate spectrum, multiplying structural matrix by seq_evectors in cuBLAS ...
        Safe_CUBLAS_Call(cublasSgemm(   handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                        gd->num_conf, iter_chunk_size, 
                                        gd->existing_contacts, // inner dimension - must be EXISTINGCONTACTS ... 
                                        &alpha, 
                                        gd->contactmap_matrix_device, gd->num_conf, 
                                        gd->seq_evector_device, gd->existing_contacts, 
                                        &beta, 
                                        gd->sfold_device, gd->num_conf));
        // get native conformation indexes and their energy levels ...
        Safe_CUDA_Call(cudaMemset(gd->efold_device, 0x77, gd->fold_chunk_size*sizeof(int)));
        grid.x = SPLIT(gd->num_conf,2*block2D.x*WORK_str); grid.y = SPLIT(iter_chunk_size,block2D.y);
        experimental_reduction2D<<<grid,block2D>>>( mutex_array, gd->sfold_device,
                                                    gd->efold_device, gd->ifold_device,
                                                    gd->num_conf, iter_chunk_size);
        // Safe_CUDA_Call(cudaDeviceSynchronize());
        // get probability of the native state ...
        // grid is exactly the same (so far) as at getting enative and e_idx_natrive ...
        // but we have to zero-out pnat array, as we're using atomics on it ...
        Safe_CUDA_Call(cudaMemset(gd->pfold_device, 0,  gd->fold_chunk_size*sizeof(float) ));
        sub_exp_reduction<<<grid,block2D>>>(    gd->sfold_device, gd->efold_device,
                                                gd->pfold_device, gd->num_conf,
                                                iter_chunk_size, fold_exp_coeff); // last parameter is 1/T
        // Safe_CUDA_Call(cudaDeviceSynchronize());
        // finally, inverse the result stored in gd->pfold_device (so far stat. summs) to get Pnat-s ...
        block2D.x = 64; block2D.y = 1;
        grid.x = SPLIT(iter_chunk_size,2*block2D.x*WORK_str); grid.y = 1;
        kernel_invert_Z<<<grid,block2D>>>(gd->pfold_device, iter_chunk_size);
        // Safe_CUDA_Call(cudaDeviceSynchronize());
        // i-th iteration is complete now: native indexes are in (gd->ifold_device) and native probabs are in (gd->pfold_device)
        // TODO make use of streams and send these partial results back to host in an asynchronous fashion ...         
        // copy partial results of Pnat back to host ...
        if (gd->outlist_fold & GALE_RETURN_PNAT_FOLD) {
            Safe_CUDA_Call(cudaMemcpy(  gd->pfold_host + i*gd->fold_chunk_size, // each chunk goes in its place @ the host ...
                                        gd->pfold_device, iter_chunk_size*sizeof(float),
                                        cudaMemcpyDeviceToHost));
        }
        // copy partial results of deltaG back to host ...
        if (gd->outlist_fold & GALE_RETURN_DG_FOLD) {
            Safe_CUDA_Call(cudaMemcpy(  gd->pfold_host + i*gd->fold_chunk_size, // each chunk goes in its place @ the host ...
                                        gd->pfold_device, iter_chunk_size*sizeof(float),
                                        cudaMemcpyDeviceToHost));
        }
        // copy partial results of native indexes back to host ...
        if (gd->outlist_fold & GALE_RETURN_INDEX_FOLD) {
            Safe_CUDA_Call(cudaMemcpy(  gd->ifold_host + i*gd->fold_chunk_size,
                                        gd->ifold_device, iter_chunk_size*sizeof(int),
                                        cudaMemcpyDeviceToHost));
        }
        // copy emins as well ...
        if (gd->outlist_fold & GALE_RETURN_EMIN_FOLD) {
            Safe_CUDA_Call(cudaMemcpy(  gd->efold_host + i*gd->fold_chunk_size,
                                        gd->efold_device, iter_chunk_size*sizeof(float),
                                        cudaMemcpyDeviceToHost));
        }
        // copy Espectrum in case you really need it ...
        if (gd->outlist_fold & GALE_RETURN_SPECTRUM_FOLD) {
            Safe_CUDA_Call(cudaMemcpy(  gd->sfold_host + i*gd->fold_chunk_size*gd->num_conf,
                                        gd->sfold_device, iter_chunk_size*gd->num_conf*sizeof(float),
                                        cudaMemcpyDeviceToHost));
        }
    }
    // dealloc locally allocated arrays ...
    Safe_CUDA_Call(cudaFree(mutex_array));
    Safe_CUBLAS_Call(cublasDestroy(handle));
    return;
}
// bind ...
void gale_bind_compute(GALE_Data *gd, float bind_temp){
    // init grid dimensions ...
    dim3 block(16,16,1);
    dim3 grid(1,1,1);
    size_t shmem = 0;
    //
    float bind_exp_coeff = -1.0f/bind_temp;
    //
    // sequences are already on the device ...
    Safe_CUDA_Call(cudaMemcpy(gd->bind_seq_pairs_device, gd->bind_seq_pairs_host, 2*gd->num_pairs*sizeof(int), cudaMemcpyHostToDevice));
    Safe_CUDA_Call(cudaMemcpy(gd->bind_conf_pairs_device, gd->bind_conf_pairs_host, 2*gd->num_pairs*sizeof(int), cudaMemcpyHostToDevice));
    //
    int i, bind_num_chunks = SPLIT(gd->num_pairs,gd->bind_chunk_size);
    fprintf(stderr, "Number of chunks is calculated in bind function:\n%d pairs to bind in %d chunks.\n",gd->num_pairs,bind_num_chunks);
    // process all chunks ...
    for (i = 0; i < bind_num_chunks; i++) {    
        // update grid & block dimensions first for every iteration ...
        block.x = 16; block.y = 16; block.z = 1; 
        grid.x = 1; grid.y = 1; grid.z = 1;
        //
        // last iter chunk mught be different than others ...
        int iter_chunk_size = (i < bind_num_chunks-1) ? (gd->bind_chunk_size) : (gd->num_pairs - i*gd->bind_chunk_size);
        //
        // arrays that goes in chunks ...
        // gd->bind_seq_pairs_device, gd->bind_conf_pairs_device
        int *seq_pairs_chunk = gd->bind_seq_pairs_device + i*2*gd->bind_chunk_size;
        int *conf_pairs_chunk = gd->bind_conf_pairs_device + i*2*gd->bind_chunk_size;
        //
        // (SeqSeqEvector is not needed ...)
        // // first kernel to generate seqseq energetic vectors ...
        // grid.x = 1; grid.y = SPLIT(iter_chunk_size,block.y); grid.z = 1;
        // shmem = ALPHABET*ALPHABET*sizeof(float) + block.y*2*gd->max_seqlen*sizeof(int);
        // // launch kernel ...
        // get_seq_pairwise_vector<<<grid,block,shmem>>>(  gd->seqarray_device, gd->seqseq_evectors_device,
        //                                                 seq_pairs_chunk, gd->forcefield_device,
        //                                                 iter_chunk_size, gd->max_seqlen);
        // // Safe_CUDA_Call(cudaDeviceSynchronize());
        // //
        // first kernel to get binding spectra coords ...
        block.x = 16; block.y = 4; block.z = 4;
        grid.x = SPLIT(gd->face_size,block.x); grid.y = 1; grid.z = SPLIT(iter_chunk_size,block.z);
        // shmem = FACES*block.z*block.x*sizeof(int);
        // // launch kernel ...
        // get_binding_spectra_coords<<<grid,block,shmem>>>(   gd->bind_residues_device, gd->bind_faces_device,
        //                                                     conf_pairs_chunk, iter_chunk_size,
        //                                                     gd->edge_len, gd->max_seqlen);
        // //
        //
        // size_t shmem_faces = FACES*block.z*block.x*sizeof(int)+FACES*ORIENTATIONS*block.z*block.x*sizeof(int);
        // size_t shmem_faces_seqs = shmem_faces + block.z*2*gd->max_seqlen*sizeof(int);
        shmem = FACES*block.z*block.x*sizeof(int)+FACES*ORIENTATIONS*block.z*block.x*sizeof(int);
        shmem += block.z*2*gd->max_seqlen*sizeof(int);
        // invoke kernel here ...
        get_binding_spectra_coords_FF<<<grid,block,shmem>>>( gd->bind_residues_device, gd->bind_faces_device, gd->seqarray_device,
                                                                        conf_pairs_chunk, seq_pairs_chunk, iter_chunk_size,
                                                                        gd->edge_len, gd->max_seqlen);
        //
        //
        // Safe_CUDA_Call(cudaDeviceSynchronize());
        //
        // get spectrum 144 for all pairs ...
        block.x = 16; block.y = 16; block.z = 1;
        grid.x = 1; grid.y = SPLIT(iter_chunk_size*C144,block.y*9); grid.z = 1;
        // shmem = gd->max_seqlen*gd->max_seqlen*sizeof(float);
        // // invoke kernel here ...
        // combine_spectra_skeleton_seqseq<<<grid,block,shmem>>>(  gd->sbind_device, gd->bind_residues_device,
        //                                                         gd->seqseq_evectors_device, iter_chunk_size,
        //                                                         gd->max_seqlen, gd->edge_len);
        //
        shmem = ALPHABET*ALPHABET*sizeof(float);
        //
        combine_spectra_skeleton_seqseq_FF<<<grid,block,shmem>>>( gd->sbind_device, gd->bind_residues_device,
                                                                                            gd->forcefield_device, iter_chunk_size,
                                                                                            gd->max_seqlen, gd->edge_len);
        //
        //
        // Safe_CUDA_Call(cudaDeviceSynchronize());
        //
        // find native states (bind states) ...
        grid.x = 1; grid.y = SPLIT(iter_chunk_size,block.y); grid.z = 1;
        min_reduction2D_144<<<grid,block>>>(gd->sbind_device, gd->ebind_device,
                                            gd->ibind_device, iter_chunk_size);
        // Safe_CUDA_Call(cudaDeviceSynchronize());
        //
        // find pbind here ...
        sub_exp_reduction144<<<grid,block>>>(   gd->sbind_device, gd->ebind_device,
                                                gd->pbind_device, iter_chunk_size, bind_exp_coeff);
        // Safe_CUDA_Call(cudaDeviceSynchronize());
        ///////////////////////////////////
        //
        // copy partial results of Pnat binding back to host ...
        if (gd->outlist_bind & GALE_RETURN_PNAT_BIND) {
            Safe_CUDA_Call(cudaMemcpy(  gd->pbind_host + i*gd->bind_chunk_size, // each chunk goes in its place @ the host ...
                                        gd->pbind_device, iter_chunk_size*sizeof(float),
                                        cudaMemcpyDeviceToHost));
        }
        // copy partial results of deltaG binding back to host ...
        if (gd->outlist_bind & GALE_RETURN_DG_BIND) {
            Safe_CUDA_Call(cudaMemcpy(  gd->pbind_host + i*gd->bind_chunk_size, // each chunk goes in its place @ the host ...
                                        gd->pbind_device, iter_chunk_size*sizeof(float),
                                        cudaMemcpyDeviceToHost));
        }
        // copy partial results of native binding indexes back to host ...
        if (gd->outlist_bind & GALE_RETURN_INDEX_BIND) {
            Safe_CUDA_Call(cudaMemcpy(  gd->ibind_host + i*gd->bind_chunk_size,
                                        gd->ibind_device, iter_chunk_size*sizeof(int),
                                        cudaMemcpyDeviceToHost));
        }
        // copy emins of binding as well ...
        if (gd->outlist_bind & GALE_RETURN_EMIN_BIND) {
            Safe_CUDA_Call(cudaMemcpy(  gd->ebind_host + i*gd->bind_chunk_size,
                                        gd->ebind_device, iter_chunk_size*sizeof(float),
                                        cudaMemcpyDeviceToHost));
        }
        // copy Espectrum 144 binding in case you really need it ...
        if (gd->outlist_bind & GALE_RETURN_SPECTRUM_BIND) {
            Safe_CUDA_Call(cudaMemcpy(  gd->sbind_host + i*gd->bind_chunk_size*C144,
                                        gd->sbind_device, iter_chunk_size*C144*sizeof(float),
                                        cudaMemcpyDeviceToHost));
        }
        //
    }    
    //
    return;
}

// deallocators ...
void gale_close(GALE_Data **gd){
    // clean up device allocations ...
    if((*gd)->seqarray_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->seqarray_device)); }
    if((*gd)->idxlut_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->idxlut_device)); }
    if((*gd)->contactmap_matrix_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->contactmap_matrix_device)); }
    if((*gd)->forcefield_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->forcefield_device)); }
    if((*gd)->seq_evector_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->seq_evector_device)); }
    if((*gd)->pfold_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->pfold_device)); }
    if((*gd)->gfold_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->gfold_device)); }
    if((*gd)->ifold_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->ifold_device)); }
    if((*gd)->efold_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->efold_device)); }  
    if((*gd)->sfold_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->sfold_device)); }
    // bind ...
    if((*gd)->bind_seq_pairs_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->bind_seq_pairs_device)); }
    if((*gd)->bind_conf_pairs_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->bind_conf_pairs_device)); }
    if((*gd)->bind_faces_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->bind_faces_device)); }
    // if((*gd)->seqseq_evectors_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->seqseq_evectors_device)); }
    if((*gd)->bind_residues_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->bind_residues_device)); }
    if((*gd)->pbind_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->pbind_device)); }
    if((*gd)->gbind_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->gbind_device)); }
    if((*gd)->ibind_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->ibind_device)); }
    if((*gd)->ebind_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->ebind_device)); }
    if((*gd)->sbind_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->sbind_device)); }
    //
    //
    //
    //
    // clean up host allocations ...
    if((*gd)->contactmap_host != NULL) { free((*gd)->contactmap_host); }
    if((*gd)->forcefield_host != NULL) { free((*gd)->forcefield_host); }
    // binding related ...
    if((*gd)->bind_faces_host != NULL) { free((*gd)->bind_faces_host); }
    //
    //
    //
    if ((*gd)->report_level >= GALE_REPORTS_DEFAULT) {fprintf(stderr, "GALE data memory deallocated\n");}
    if ((*gd)->report_level >= GALE_REPORTS_VERBOSE) {
        size_t free_device_mem, total_device_mem;
        Safe_CUDA_Call(cudaMemGetInfo(&free_device_mem, &total_device_mem));
        printf("device memory: %zu bytes are free after deallocation (total %zu)\n", free_device_mem, total_device_mem);
    }
    //
    //
    // deallocate the structure itself ...
    if((*gd) != NULL) { free((*gd)); }
    //
    return;
}










