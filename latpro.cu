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




void Check_CUDA_Error(const char *message){
    cudaError_t error = cudaGetLastError();
    if(error!=cudaSuccess) {
        fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
        exit(1);
    }                        
}


// assigns NULLs to every pointer in a GALE data struct ...
void gale_init_null(GALE_Data **gd){
    // allocating struct itself first ...
    (*gd) = (GALE_Data *)malloc(sizeof(GALE_Data));
    //
    (*gd)->contactmap_host = NULL;
    (*gd)->forcefield_host = NULL;
    (*gd)->seqarray_host = NULL;
    (*gd)->pfold_host = NULL; 
    (*gd)->ifold_host = NULL;
    // binding related ...
    (*gd)->bind_seq_pairs_host = NULL;
    (*gd)->bind_conf_pairs_host = NULL;
    (*gd)->bind_faces_host = NULL;
    (*gd)->pbind_host = NULL;
    (*gd)->ebind_host = NULL;
    (*gd)->ibind_host = NULL;
    //
    //DEVICE
    // general & folding related ...
    (*gd)->seqarray_device = NULL;
    (*gd)->idxlut_device = NULL;
    (*gd)->contactmap_matrix_device = NULL;
    (*gd)->forcefield_device = NULL;
    (*gd)->seq_evector_device = NULL;
    (*gd)->sfold_device = NULL;
    (*gd)->pfold_device = NULL; 
    (*gd)->efold_device = NULL; 
    (*gd)->ifold_device = NULL;
    // binding related ...
    (*gd)->bind_seq_pairs_device = NULL;
    (*gd)->bind_conf_pairs_device = NULL;
    (*gd)->bind_faces_device = NULL;
    (*gd)->seqseq_evectors_device = NULL;
    // size_t binding_residues_size = num_pairs*C144*edge_squared*sizeof(int);
    (*gd)->bind_residues_device = NULL;
    (*gd)->sbind_device = NULL;
    (*gd)->ebind_device = NULL;
    (*gd)->pbind_device = NULL;
    (*gd)->ibind_device = NULL;
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
    //
    return;
}
// loading force field, along with allocating it on the device ...
void gale_load_forcefield(GALE_Data *gd, const char *fname){
    //
    int i, aa1, aa2; 
    float e, esum = 0.0f;
    size_t forcefield_size = ALPHABET*ALPHABET*sizeof(float);
    // allocating host and device forcefiled arrays ...
    gd->forcefield_host = (float *)malloc(forcefield_size);
    assert(gd->forcefield_host != NULL);
    // 
    FILE *fp = fopen(fname,"r"); 
    if (fp == NULL) { fprintf(stderr, "cannot read %s\n",fname); exit(1); } 
    //
    for(i=0; i < ALPHABET*ALPHABET; i++) {
        int matched = fscanf(fp, "%d %d %f", &aa1, &aa2, &e); 
        assert(matched == 3);
        gd->forcefield_host[aa1 + ALPHABET*aa2] = e; 
        esum += e; 
    } 
    esum /= (ALPHABET*ALPHABET); 
    // normalize to zero mean to avoid exponent underflows in partition function 
    // for(i=0;i<ALPHABET*ALPHABET;i++) { gd->forcefield_host[i] -= esum; } 
    //printf("read energies from %s\n", fname); 
    fclose(fp);
    fprintf(stderr, "Read energies from %s\n", fname);
    // device allocation and copy ...
    Safe_CUDA_Call(cudaMalloc((void**)&gd->forcefield_device, forcefield_size ));
    Safe_CUDA_Call(cudaMemcpy(gd->forcefield_device, gd->forcefield_host, forcefield_size, cudaMemcpyHostToDevice));
    // forcefield copied to device memory ...
    return;
}
// auto load mode so far ...
void gale_load_contactmap(GALE_Data *gd, const char *fname){
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
    gd->contactmap_host = (unsigned char *)malloc(2 * gd->max_numcontacts * gd->num_conf *sizeof(unsigned char));
    assert(gd->contactmap_host != NULL);
    unsigned char * MaxContBufferA = (unsigned char *)malloc(gd->max_numcontacts*sizeof(unsigned char));
    assert(MaxContBufferA != NULL);
    unsigned char * MaxContBufferB = (unsigned char *)malloc(gd->max_numcontacts*sizeof(unsigned char));
    assert(MaxContBufferB != NULL);
    //
    //
    printf("File with contacts is read: %d<=NumberOfContacts<=%d, NumberOfConformations=%d \n",LocalMinNumOfConts,LocalMaxNumOfConts,gd->num_conf);
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
    //
    fprintf(stderr,"Using given contact maps, estimated max sequence length is %d\n",max_seqlen_conf_estimate);
    fprintf(stderr,"Absolute max number of contacts based on estimated sequence length is %d\n",abs_max_contacts);
    fprintf(stderr,"There are %d contacts exist in all given %d conformations: contacts sparsity %.3f\n",gd->existing_contacts,gd->num_conf,contacts_sparsity);
    //
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
    printf("contactmap array converted into zero-one form suitable for device, and copied to device.\n");
    free(layover_contactmap_local);
    free(contactmap_matrix_host_local);
    //
    return;
    //
    //
}
// auto load mode so far, handles cubic proteins only ...
void gale_load_bind_faces(GALE_Data *gd, const char *fname){
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

// WORKSIZES ...
// size of the chunks to be processed as 1 grid on a device ...
void gale_set_fold_worksize(GALE_Data *gd, int fold_worksize){
    gd->fold_chunk_size = fold_worksize;
    fprintf(stderr, "Number of seq to fold on GPU at the same time is set: %d\n", gd->fold_chunk_size);
    //
    return;
}
// size of the chunks to be processed as 1 grid on a device ...
void gale_set_bind_worksize(GALE_Data *gd, int bind_worksize){
    gd->bind_chunk_size = bind_worksize;
    fprintf(stderr, "Number of seq pairs to bind on GPU at the same time is set: %d\n", gd->bind_chunk_size);
    //
    return;
}
// WORKSIZES ...


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
// set (assign) host pnat array ...
void gale_set_pfold(GALE_Data *gd, float *pfold_array){
    gd->pfold_host = pfold_array;
    return;
}
// set (assign) host native indexes array ...
void gale_set_ifold(GALE_Data *gd, int *ifold_array){
    gd->ifold_host = ifold_array;
    return;
}
// set (assign) host pnat array ...
void gale_set_pbind(GALE_Data *gd, float *pbind_array){
    gd->pbind_host = pbind_array;
    return;
}
// set (assign) host native indexes array ...
void gale_set_ibind(GALE_Data *gd, int *ibind_array){
    gd->ibind_host = ibind_array;
    return;
}
// set (assign) host native emin array ...
void gale_set_ebind(GALE_Data *gd, float *ebind_array){
    gd->ebind_host = ebind_array;
    return;
}

// prepare functions to perform device allocations ...
void gale_fold_prepare(GALE_Data *gd){
    //
    Safe_CUDA_Call(cudaMalloc((void**)&gd->seqarray_device, gd->num_seq*gd->max_seqlen*sizeof(unsigned char) ));
    //
    // knowing chunks size ...
    // allocate device arrays that rely on the chunk_size here ...
    // work or chunk size is to replace num_seq or num_pairs to be processed per batch ...
    Safe_CUDA_Call(cudaMalloc((void **)&gd->seq_evector_device, gd->fold_chunk_size*gd->existing_contacts*sizeof(float) ));
    Safe_CUDA_Call(cudaMalloc((void **)&gd->sfold_device, gd->fold_chunk_size*gd->num_conf*sizeof(float) ));
    Safe_CUDA_Call(cudaMalloc((void **)&gd->pfold_device, gd->fold_chunk_size*sizeof(float) ));
    Safe_CUDA_Call(cudaMalloc((void **)&gd->efold_device, gd->fold_chunk_size*sizeof(float) ));
    Safe_CUDA_Call(cudaMalloc((void **)&gd->ifold_device, gd->fold_chunk_size*sizeof(int)));
    //
    return;
}
void gale_bind_prepare(GALE_Data *gd){
    //
    Safe_CUDA_Call(cudaMalloc((void**)&gd->bind_seq_pairs_device, 2*gd->num_pairs*sizeof(int) ));
    Safe_CUDA_Call(cudaMalloc((void**)&gd->bind_conf_pairs_device, 2*gd->num_pairs*sizeof(int) ));
    //
    // allocate device arrays that rely on the chunk_size here ...
    // work or chunk size is to replace num_seq or num_pairs to be processed per batch ...
    Safe_CUDA_Call(cudaMalloc((void **)&gd->seqseq_evectors_device, gd->bind_chunk_size*gd->max_seqlen*gd->max_seqlen*sizeof(float)));
    Safe_CUDA_Call(cudaMalloc((void **)&gd->bind_residues_device, gd->bind_chunk_size*C144*gd->face_size*sizeof(int)));
    Safe_CUDA_Call(cudaMalloc((void **)&gd->sbind_device, gd->bind_chunk_size*C144*sizeof(float)));
    Safe_CUDA_Call(cudaMalloc((void **)&gd->ebind_device, gd->bind_chunk_size*sizeof(float)));
    Safe_CUDA_Call(cudaMalloc((void **)&gd->pbind_device, gd->bind_chunk_size*sizeof(float)));
    Safe_CUDA_Call(cudaMalloc((void **)&gd->ibind_device, gd->bind_chunk_size*sizeof(int)));
    //
    return;
}

// fold ...
void gale_fold_compute(GALE_Data *gd){
    // folding in chuncks ...
    // allocations for cuBLAS ...
    cublasHandle_t handle;
    Safe_CUBLAS_Call(cublasCreate(&handle));
    const float alpha = 1.0f;
    const float beta  = 0.0f;
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
    fprintf(stderr, "Number of chunks is calculated in fold function:\n%d sequences to fold in %d chunks.\n",gd->num_seq,fold_num_chunks);
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
        // Check_CUDA_Error("Converting seqarray to energetic vectors failed ...");
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
        // Check_CUDA_Error("Reduction of spectrum to get native conformation indexes and their energy levels failed ...");
        // get probability of the native state ...
        // grid is exactly the same (so far) as at getting enative and e_idx_natrive ...
        // but we have to zero-out pnat array, as we're using atomics on it ...
        Safe_CUDA_Call(cudaMemset(gd->pfold_device, 0,  gd->fold_chunk_size*sizeof(float) ));
        sub_exp_reduction<<<grid,block2D>>>(    gd->sfold_device, gd->efold_device,
                                                gd->pfold_device, gd->num_conf,
                                                iter_chunk_size, -1.25f); // last parameter is 1/T
        // Safe_CUDA_Call(cudaDeviceSynchronize());
        // Check_CUDA_Error("Reduction of spectrum to get probabilities of native states failed ...");
        // finally, inverse the result stored in gd->pfold_device (so far stat. summs) to get Pnat-s ...
        block2D.x = 64; block2D.y = 1;
        grid.x = SPLIT(iter_chunk_size,2*block2D.x*WORK_str); grid.y = 1;
        kernel_invert_Z<<<grid,block2D>>>(gd->pfold_device, iter_chunk_size);
        // Safe_CUDA_Call(cudaDeviceSynchronize());
        // Check_CUDA_Error("Inverting stat. summs failed ...");
        // i-th iteration is complete now: native indexes are in (gd->ifold_device) and native probabs are in (gd->pfold_device)
        // TODO make use of streams and send these partial results back to host in an asynchronous fashion ...         
        // copy partial results of Pnat back to host ...
        Safe_CUDA_Call(cudaMemcpy(  gd->pfold_host + i*gd->fold_chunk_size, // each chunk goes in its place @ the host ...
                                    gd->pfold_device, iter_chunk_size*sizeof(float),
                                    cudaMemcpyDeviceToHost));
        // copy partial results of native indexes back to host ...
        Safe_CUDA_Call(cudaMemcpy(  gd->ifold_host + i*gd->fold_chunk_size,
                                    gd->ifold_device, iter_chunk_size*sizeof(int),
                                    cudaMemcpyDeviceToHost));
    }
    // dealloc locally allocated arrays ...
    Safe_CUDA_Call(cudaFree(mutex_array));
    Safe_CUBLAS_Call(cublasDestroy(handle));
    return;
}
// bind ...
void gale_bind_compute(GALE_Data *gd){
    // init grid dimensions ...
    dim3 block(16,16,1);
    dim3 grid(1,1,1);
    size_t shmem = 0;
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
        //
        // first kernel to generate seqseq energetic vectors ...
        grid.x = 1; grid.y = SPLIT(iter_chunk_size,block.y); grid.z = 1;
        shmem = ALPHABET*ALPHABET*sizeof(float) + block.y*2*gd->max_seqlen*sizeof(int);
        // launch kernel ...
        get_seq_pairwise_vector<<<grid,block,shmem>>>(  gd->seqarray_device, gd->seqseq_evectors_device,
                                                        seq_pairs_chunk, gd->forcefield_device,
                                                        iter_chunk_size, gd->max_seqlen);
        // Safe_CUDA_Call(cudaDeviceSynchronize());
        // Check_CUDA_Error("seqseq matrix failed ...");
        //
        // secind kernel to get binding spectra coords ...
        block.x = 16; block.y = 4; block.z = 4;
        grid.x = SPLIT(gd->face_size,block.x); grid.y = 1; grid.z = SPLIT(iter_chunk_size,block.z);
        shmem = FACES*block.z*block.x*sizeof(int);
        // launch kernel ...
        get_binding_spectra_coords<<<grid,block,shmem>>>(   gd->bind_residues_device, gd->bind_faces_device,
                                                            conf_pairs_chunk, iter_chunk_size,
                                                            gd->edge_len, gd->max_seqlen);
        // Safe_CUDA_Call(cudaDeviceSynchronize());
        // Check_CUDA_Error("getting binding residues failed ...");
        //
        // get spectrum 144 for all pairs ...
        block.x = 16; block.y = 16; block.z = 1;
        grid.x = 1; grid.y = SPLIT(iter_chunk_size*C144,block.y*9); grid.z = 1;
        shmem = gd->max_seqlen*gd->max_seqlen*sizeof(float);
        // invoke kernel here ...
        combine_spectra_skeleton_seqseq<<<grid,block,shmem>>>(  gd->sbind_device, gd->bind_residues_device,
                                                                gd->seqseq_evectors_device, iter_chunk_size,
                                                                gd->max_seqlen, gd->edge_len);
        // Safe_CUDA_Call(cudaDeviceSynchronize());
        // Check_CUDA_Error("getting binding spectras failed ...");
        //
        // find native states (bind states) ...
        grid.x = 1; grid.y = SPLIT(iter_chunk_size,block.y); grid.z = 1;
        min_reduction2D_144<<<grid,block>>>(gd->sbind_device, gd->ebind_device,
                                            gd->ibind_device, iter_chunk_size);
        // Safe_CUDA_Call(cudaDeviceSynchronize());
        // Check_CUDA_Error("min reduction binding failed");
        //
        // find pbind here ...
        sub_exp_reduction144<<<grid,block>>>(   gd->sbind_device, gd->ebind_device,
                                                gd->pbind_device, iter_chunk_size, -1.25f);
        // Safe_CUDA_Call(cudaDeviceSynchronize());
        // Check_CUDA_Error("Failed to launch kernel sub-e BIND");
        ///////////////////////////////////
        Safe_CUDA_Call(cudaMemcpy(  gd->pbind_host + i*gd->bind_chunk_size,
                                    gd->pbind_device, iter_chunk_size*sizeof(float),
                                    cudaMemcpyDeviceToHost));
        Safe_CUDA_Call(cudaMemcpy(  gd->ebind_host + i*gd->bind_chunk_size,
                                    gd->ebind_device, iter_chunk_size*sizeof(float),
                                    cudaMemcpyDeviceToHost));
        Safe_CUDA_Call(cudaMemcpy(  gd->ibind_host + i*gd->bind_chunk_size,
                                    gd->ibind_device, iter_chunk_size*sizeof(int),
                                    cudaMemcpyDeviceToHost));
    }    
    //
    return;
}

// deallocators ...
void gale_free_fold_arrays(GALE_Data *gd){
    // clean up folding allocations ...
    if (gd->seqarray_device != NULL) { Safe_CUDA_Call(cudaFree(gd->seqarray_device)); }
    if (gd->seq_evector_device != NULL) { Safe_CUDA_Call(cudaFree(gd->seq_evector_device)); }
    if (gd->sfold_device != NULL) { Safe_CUDA_Call(cudaFree(gd->sfold_device)); }
    if (gd->pfold_device != NULL) { Safe_CUDA_Call(cudaFree(gd->pfold_device)); }
    if (gd->efold_device != NULL) { Safe_CUDA_Call(cudaFree(gd->efold_device)); }
    if (gd->ifold_device != NULL) { Safe_CUDA_Call(cudaFree(gd->ifold_device)); }
    //
    return;
}
void gale_free_bind_arrays(GALE_Data *gd){
    // clean up binding allocations ...
    if (gd->bind_seq_pairs_device != NULL) { Safe_CUDA_Call(cudaFree(gd->bind_seq_pairs_device)); }
    if (gd->bind_conf_pairs_device != NULL) { Safe_CUDA_Call(cudaFree(gd->bind_conf_pairs_device)); }
    if (gd->seqseq_evectors_device != NULL) { Safe_CUDA_Call(cudaFree(gd->seqseq_evectors_device)); }
    if (gd->bind_residues_device != NULL) { Safe_CUDA_Call(cudaFree(gd->bind_residues_device)); }    
    if (gd->sbind_device != NULL) { Safe_CUDA_Call(cudaFree(gd->sbind_device)); }
    if (gd->ebind_device != NULL) { Safe_CUDA_Call(cudaFree(gd->ebind_device)); }
    if (gd->pbind_device != NULL) { Safe_CUDA_Call(cudaFree(gd->pbind_device)); }
    if (gd->ibind_device != NULL) { Safe_CUDA_Call(cudaFree(gd->ibind_device)); }
    //
    return;
}
void gale_close(GALE_Data **gd){
    fprintf(stderr, "Deallocating data structures ...\n");
    //
    // clean up device allocations ...
    if((*gd)->seqarray_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->seqarray_device)); }
    if((*gd)->idxlut_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->idxlut_device)); }
    if((*gd)->contactmap_matrix_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->contactmap_matrix_device)); }
    if((*gd)->forcefield_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->forcefield_device)); }
    if((*gd)->seq_evector_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->seq_evector_device)); }
    if((*gd)->sfold_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->sfold_device)); }
    if((*gd)->pfold_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->pfold_device)); }
    if((*gd)->efold_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->efold_device)); }  
    if((*gd)->ifold_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->ifold_device)); }
    // bind ...
    if((*gd)->bind_seq_pairs_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->bind_seq_pairs_device)); }
    if((*gd)->bind_conf_pairs_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->bind_conf_pairs_device)); }
    if((*gd)->bind_faces_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->bind_faces_device)); }
    if((*gd)->seqseq_evectors_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->seqseq_evectors_device)); }
    if((*gd)->bind_residues_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->bind_residues_device)); }
    if((*gd)->sbind_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->sbind_device)); }
    if((*gd)->ebind_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->ebind_device)); }
    if((*gd)->pbind_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->pbind_device)); }
    if((*gd)->ibind_device != NULL) { Safe_CUDA_Call(cudaFree((*gd)->ibind_device)); }
    //
    // clean up host allocations ...
    if((*gd)->contactmap_host != NULL) { free((*gd)->contactmap_host); }
    if((*gd)->forcefield_host != NULL) { free((*gd)->forcefield_host); }
    // binding related ...
    if((*gd)->bind_faces_host != NULL) { free((*gd)->bind_faces_host); }
    //
    // deallocate the structure itself ...
    if((*gd) != NULL) { free((*gd)); }
    //
    fprintf(stderr, "Deallocation completed.\n");
    //
    return;
}










