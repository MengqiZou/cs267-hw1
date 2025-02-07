#include <immintrin.h>

const char* dgemm_desc = "Simple blocked dgemm.";

// block size targeted to L2 cache
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 28
#endif

// block size of micro-kernel targeted to L1 cache
#ifndef MICRO_BLOCK_SIZE
#define MICRO_BLOCK_SIZE 28
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.

static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
    // For each row i of A
    for (int i = 0; i < M; ++i) {
        // For each column j of B
        for (int j = 0; j < N; ++j) {
            // Compute C(i,j)
            double cij = C[i + j * lda];
            for (int k = 0; k < K; ++k) {
                cij += A[i + k * lda] * B[k + j * lda];
            }
            C[i + j * lda] = cij;
        }
    }
}
*/

/*
 * Micro-kernel for the inner-most block multiplication - with ijk ordering

 static void do_micro_kernel(int lda, int M, int N, int K, double* A, double* B, double* C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double cij = C[i + j * lda];
            for (int k = 0; k < K; ++k) {
                cij += A[i + k * lda] * B[k + j * lda];
            }
            C[i + j * lda] = cij;
        }
    }
}
*/

/*
 micro-kernel with kij ordering
 
static void do_micro_kernel(int lda, int M, int N, int K, double* A, double* B, double* C) {
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                C[i + j * lda] += A[i + k * lda] * B[k + j * lda];
            }
        }
    }
}
*/

/*
// NOT WORKING: SEGMENTATION FAULT - SIMD-optimized micro-kernel using FMA
void do_micro_kernel(int lda, int M, int N, int K, double* A, double* B, double* C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            __m256d cij = _mm256_load_pd(&C[i + j * lda]); // Load the initial value of C[i, j] into a SIMD register

            // Loop over K to perform the dot product
            for (int k = 0; k < K; ++k) {
                __m256d Ar = _mm256_set1_pd(A[i + k * lda]);  // Broadcast A[i, k] to all elements of a register
                __m256d Br = _mm256_broadcast_sd(&B[k + j * lda]); // Broadcast B[k, j] to all elements of a register

                // Perform FMA: cij += A[i, k] * B[k, j]
                cij = _mm256_fmadd_pd(Ar, Br, cij);
            }

            // Store the final result in C[i,j] by summing the 4 elements of cij
            double temp[4];  // Temporary array to store the result from cij
            _mm256_store_pd(temp, cij);  // Store the values from cij to the temporary array

            // Sum the 4 elements of cij to accumulate the result for C[i,j]
            C[i + j * lda] += temp[0] + temp[1] + temp[2] + temp[3];
        }
    }
}
*/

/*
 * Two-level blocking strategy: first level is targeting L2, second level is targeting L1
 */
static void do_block_with_micro_kernel(int lda, int M, int N, int K, double* A, double* B, double* C) {
    for (int i = 0; i < M; i += MICRO_BLOCK_SIZE) {
        for (int j = 0; j < N; j += MICRO_BLOCK_SIZE) {
            for (int k = 0; k < K; k += MICRO_BLOCK_SIZE) {
                // dimensions of micro-block
                int micro_M = min(MICRO_BLOCK_SIZE, M - i);
                int micro_N = min(MICRO_BLOCK_SIZE, N - j);
                int micro_K = min(MICRO_BLOCK_SIZE, K - k);

                // call micro-kernel for small block multiplication
                do_micro_kernel(lda, micro_M, micro_N, micro_K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
            }
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
                do_block_with_micro_kernel(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
            }
        }
    }
}