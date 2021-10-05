#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "likwid-stuff.h"

// #define MY_MARKER_REGION_NAME "myMarkerName"

const char* dgemm_desc = "Blocked dgemm, OpenMP-enabled";

void copy_to_block(double *, int, int, int, double *, int);
void copy_from_block(double *, int, int, int, double *, int);
void square_dgemm(int, double*, double*, double*);

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm_blocked(int n, int block_size, double* A, double* B, double* C) 
{
   // insert your code here: implementation of blocked matrix multiply with copy optimization and OpenMP parallelism enabled

   // be sure to include LIKWID_MARKER_START(MY_MARKER_REGION_NAME) inside the block of parallel code,
   // but before your matrix multiply code, and then include LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME)
   // after the matrix multiply code but before the end of the parallel code block.
   std::cout << "Insert your blocked matrix multiply with copy optimization, openmp-parallel edition here " << std::endl;
   int nblocks = n/block_size;
        
   double * Alocal = (double*) malloc(block_size * block_size * sizeof(double));
   double * Blocal = (double*) malloc(block_size * block_size * sizeof(double));
   double * Clocal = (double*) malloc(block_size * block_size * sizeof(double));

   #pragma omp parallel
   {
   LIKWID_MARKER_START(MY_MARKER_REGION_NAME);
   #pragma omp parallel for collapse(2)
   for (int i = 0; i < nblocks; i++){
      for (int j = 0; j < nblocks; j++){ 
         //copy from C[i*bs, j*bs] into Clocal
         copy_to_block(C, n, i * block_size, j * block_size, Clocal, block_size);
         #pragma omp parallel for
         for(int k = 0; k < nblocks; k++){ 
            //copy from A[i*bs, k*bs] into Alocal
            copy_to_block(A, n, i * block_size, k * block_size, Alocal, block_size);
            //copy from B[k*bs, j*bs] into Blocal
            copy_to_block(B, n, k * block_size, j * block_size, Blocal, block_size);
            
            square_dgemm(block_size, Alocal, Blocal, Clocal);
            // for (int ii=0; ii<block_size; ii++){
            //    for (int jj=0; jj<block_size; jj++){
            //       double temp = Clocal[ii + jj * block_size];
            //       for(int kk=0; kk<block_size; kk++){
            //          // C[i,j] += A[i,k] * B[k,j]
            //          temp += Alocal[ii + kk * block_size] * Blocal[kk + jj * block_size];
            //       }
            //       Clocal[ii + jj * block_size] = temp;
            //    }
         
            // copy from Clocal back to  C[i*bs, j*bs]
            copy_from_block(Clocal, n, i * block_size, j * block_size, C, block_size);
         }
      }
   }
   LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME);
   free(Alocal);
   free(Blocal);
   free(Clocal);
   }
}

void copy_to_block(double *src_matrix, int n, int ioffset, int joffset, double *dst_block, int block_size)
{
    int src_offset = joffset * n + ioffset;
    int dst_offset = 0;
    for (int i = 0; i < block_size; i++, src_offset += n, dst_offset += block_size)
         memcpy(dst_block + dst_offset, src_matrix + src_offset, sizeof(double)*block_size);
}

void copy_from_block(double *src_block, int n, int ioffset, int joffset, double *dst_matrix, int block_size)
{
    int src_offset = 0;
    int dst_offset = joffset * n + ioffset;
    for (int i = 0; i < block_size; i++, src_offset += block_size, dst_offset += n)
         memcpy(dst_matrix + dst_offset, src_block + src_offset, sizeof(double)*block_size);
}

// copy from dgemm-basic

void square_dgemm(int n, double* A, double* B, double* C) 
{
   // #pragma omp parallel for
   for (int i=0; i<n; i++){
      for (int j=0; j<n; j++){
         double temp = C[i + j * n];
         for(int k=0; k<n; k++){
            // C[i,j] += A[i,k] * B[k,j]
            temp += A[i + k * n] * B[k + j * n];
         }
         C[i + j * n] = temp;
      }
   }
}