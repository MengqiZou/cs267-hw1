#include <immintrin.h>

const char* dgemm_desc = "simd ijk block1 32 block2 128 j+=8 vector256.";

#ifndef BLOCK_SIZE1
#define BLOCK_SIZE1 32
#endif


#ifndef BLOCK_SIZE2
#define BLOCK_SIZE2 128
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

    // **AVX-optimized loop processing 8 columns at a time**
    for (i = 0; i + 4 <= M; i += 4) {  
        for (j = 0; j + 8 <= N; j += 8) {  

            // Load C(i:i+4, j:j+8)
            __m256d C0 = _mm256_loadu_pd(&C[i + (j + 0) * lda]);
            __m256d C1 = _mm256_loadu_pd(&C[i + (j + 1) * lda]);
            __m256d C2 = _mm256_loadu_pd(&C[i + (j + 2) * lda]);
            __m256d C3 = _mm256_loadu_pd(&C[i + (j + 3) * lda]);
            __m256d C4 = _mm256_loadu_pd(&C[i + (j + 4) * lda]);
            __m256d C5 = _mm256_loadu_pd(&C[i + (j + 5) * lda]);
            __m256d C6 = _mm256_loadu_pd(&C[i + (j + 6) * lda]);
            __m256d C7 = _mm256_loadu_pd(&C[i + (j + 7) * lda]);

            for (k = 0; k < K; ++k) {  
                __m256d A0 = _mm256_loadu_pd(&A[i + k * lda]);

                __m256d B0 = _mm256_set1_pd(B[k + (j + 0) * lda]);
                __m256d B1 = _mm256_set1_pd(B[k + (j + 1) * lda]);
                __m256d B2 = _mm256_set1_pd(B[k + (j + 2) * lda]);
                __m256d B3 = _mm256_set1_pd(B[k + (j + 3) * lda]);
                __m256d B4 = _mm256_set1_pd(B[k + (j + 4) * lda]);
                __m256d B5 = _mm256_set1_pd(B[k + (j + 5) * lda]);
                __m256d B6 = _mm256_set1_pd(B[k + (j + 6) * lda]);
                __m256d B7 = _mm256_set1_pd(B[k + (j + 7) * lda]);

                C0 = _mm256_fmadd_pd(A0, B0, C0);
                C1 = _mm256_fmadd_pd(A0, B1, C1);
                C2 = _mm256_fmadd_pd(A0, B2, C2);
                C3 = _mm256_fmadd_pd(A0, B3, C3);
                C4 = _mm256_fmadd_pd(A0, B4, C4);
                C5 = _mm256_fmadd_pd(A0, B5, C5);
                C6 = _mm256_fmadd_pd(A0, B6, C6);
                C7 = _mm256_fmadd_pd(A0, B7, C7);
            }

            _mm256_storeu_pd(&C[i + (j + 0) * lda], C0);
            _mm256_storeu_pd(&C[i + (j + 1) * lda], C1);
            _mm256_storeu_pd(&C[i + (j + 2) * lda], C2);
            _mm256_storeu_pd(&C[i + (j + 3) * lda], C3);
            _mm256_storeu_pd(&C[i + (j + 4) * lda], C4);
            _mm256_storeu_pd(&C[i + (j + 5) * lda], C5);
            _mm256_storeu_pd(&C[i + (j + 6) * lda], C6);
            _mm256_storeu_pd(&C[i + (j + 7) * lda], C7);
        }
    }

    // **Handle M % 4 remaining rows**
    for (; i < M; i++) {  
        for (j = 0; j + 8 <= N; j += 8) {
            double C0 = C[i + (j + 0) * lda];
            double C1 = C[i + (j + 1) * lda];
            double C2 = C[i + (j + 2) * lda];
            double C3 = C[i + (j + 3) * lda];
            double C4 = C[i + (j + 4) * lda];
            double C5 = C[i + (j + 5) * lda];
            double C6 = C[i + (j + 6) * lda];
            double C7 = C[i + (j + 7) * lda];

            for (k = 0; k < K; ++k) {
                double A0 = A[i + k * lda];

                double B0 = B[k + (j + 0) * lda];
                double B1 = B[k + (j + 1) * lda];
                double B2 = B[k + (j + 2) * lda];
                double B3 = B[k + (j + 3) * lda];
                double B4 = B[k + (j + 4) * lda];
                double B5 = B[k + (j + 5) * lda];
                double B6 = B[k + (j + 6) * lda];
                double B7 = B[k + (j + 7) * lda];

                C0 += A0 * B0;
                C1 += A0 * B1;
                C2 += A0 * B2;
                C3 += A0 * B3;
                C4 += A0 * B4;
                C5 += A0 * B5;
                C6 += A0 * B6;
                C7 += A0 * B7;
            }

            C[i + (j + 0) * lda] = C0;
            C[i + (j + 1) * lda] = C1;
            C[i + (j + 2) * lda] = C2;
            C[i + (j + 3) * lda] = C3;
            C[i + (j + 4) * lda] = C4;
            C[i + (j + 5) * lda] = C5;
            C[i + (j + 6) * lda] = C6;
            C[i + (j + 7) * lda] = C7;
        }
    }

    // **Handle remaining columns**
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

  for (int block_k = 0; block_k < lda; block_k += BLOCK_SIZE2) 
{

  for (int block_j = 0; block_j < lda; block_j += BLOCK_SIZE2) 
  {

      for (int block_i = 0; block_i < lda; block_i += BLOCK_SIZE2) 
      {

          int limit_k = block_k + min(BLOCK_SIZE2, lda - block_k);
          int limit_j = block_j + min(BLOCK_SIZE2, lda - block_j);
          int limit_i = block_i + min(BLOCK_SIZE2, lda - block_i);

        for (int k = block_k; k < limit_k; k += BLOCK_SIZE1) 
        {

          for (int j = block_j; j < limit_j; j += BLOCK_SIZE1) 
          {

            for (int i = block_i; i < limit_i; i += BLOCK_SIZE1) 
            {
      
              int K = min(BLOCK_SIZE1, limit_k - k);
              int N = min(BLOCK_SIZE1, limit_j - j);
              int M = min(BLOCK_SIZE1, limit_i - i);
              do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
            }
          }
        }
      }
    }
  }
}
