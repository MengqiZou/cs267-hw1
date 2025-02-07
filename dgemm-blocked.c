#include <immintrin.h>

const char* dgemm_desc = "simd ijk blocksize32 vector256.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
 //cache = 

static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
    int i = 0, j = 0, k = 0;


    for (i = 0; i + 4 <= M; i += 4) { 
        for (j = 0; j + 4 <= N; j += 4) { 
    
            __m256d C0 = _mm256_loadu_pd(&C[i + (j + 0) * lda]);
            __m256d C1 = _mm256_loadu_pd(&C[i + (j + 1) * lda]);
            __m256d C2 = _mm256_loadu_pd(&C[i + (j + 2) * lda]);
            __m256d C3 = _mm256_loadu_pd(&C[i + (j + 3) * lda]);

            for (k = 0; k < K; ++k) {  
                __m256d A0 = _mm256_loadu_pd(&A[i + k * lda]);

                __m256d B0 = _mm256_set1_pd(B[k + (j + 0) * lda]);
                __m256d B1 = _mm256_set1_pd(B[k + (j + 1) * lda]);
                __m256d B2 = _mm256_set1_pd(B[k + (j + 2) * lda]);
                __m256d B3 = _mm256_set1_pd(B[k + (j + 3) * lda]);

                C0 = _mm256_fmadd_pd(A0, B0, C0);
                C1 = _mm256_fmadd_pd(A0, B1, C1);
                C2 = _mm256_fmadd_pd(A0, B2, C2);
                C3 = _mm256_fmadd_pd(A0, B3, C3);
            }

            _mm256_storeu_pd(&C[i + (j + 0) * lda], C0);
            _mm256_storeu_pd(&C[i + (j + 1) * lda], C1);
            _mm256_storeu_pd(&C[i + (j + 2) * lda], C2);
            _mm256_storeu_pd(&C[i + (j + 3) * lda], C3);
        }
    }

    for (; i < M; i++) { 
        for (j = 0; j + 4 <= N; j += 4) {
            double C0 = C[i + (j + 0) * lda];
            double C1 = C[i + (j + 1) * lda];
            double C2 = C[i + (j + 2) * lda];
            double C3 = C[i + (j + 3) * lda];

            for (k = 0; k < K; ++k) {
                double A0 = A[i + k * lda];

                double B0 = B[k + (j + 0) * lda];
                double B1 = B[k + (j + 1) * lda];
                double B2 = B[k + (j + 2) * lda];
                double B3 = B[k + (j + 3) * lda];

                C0 += A0 * B0;
                C1 += A0 * B1;
                C2 += A0 * B2;
                C3 += A0 * B3;
            }

            C[i + (j + 0) * lda] = C0;
            C[i + (j + 1) * lda] = C1;
            C[i + (j + 2) * lda] = C2;
            C[i + (j + 3) * lda] = C3;
        }
    }

    for (; j < N; ++j) { 
        for (i = 0; i < M; ++i) {
            double sum = C[i + j * lda];
            for (k = 0; k < K; ++k) {
                sum += A[i + k * lda] * B[k + j * lda];
            }
            C[i + j * lda] = sum;
        }
    }
}






/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A, double* B, double* C) {
    // For each block-row of A
    for (int i = 0; i < lda; i += BLOCK_SIZE) {
        // For each block-column of B
        for (int j = 0; j < lda; j += BLOCK_SIZE) {
            // Accumulate block dgemms into block of C
            for (int k = 0; k < lda; k += BLOCK_SIZE) {
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min(BLOCK_SIZE, lda - i);
                int N = min(BLOCK_SIZE, lda - j);
                int K = min(BLOCK_SIZE, lda - k);
                // Perform individual block dgemm
                do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
            }
        }
    }
}
