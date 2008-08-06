#include <stdlib.h>
#include <stdio.h>

#include "dw_cholesky.h"

#include <cblas.h>

int main(int argc, char **argv)
{
	/* create a simple definite positive symetric matrix example
	 *
	 *	Hilbert matrix : h(i,j) = 1/(i+j+1)
	 * */
	unsigned size = 24*1024;
	unsigned nblocks = 24;
	unsigned pinned = 1;

	float *mat;

	mat = malloc(size*size*sizeof(float));
	initialize_system(&mat, size, pinned);

	unsigned i,j;
	for (j = 0; j < size; j++)
	{
		for (i = 0; i < size; i++)
		{
			mat[i +j*size] = (1.0f/(1.0f+i+j)) + ((i == j)?1.0f*size:0.0f);
		}
	}

//
//	printf("Input :\n");
//
//	for (j = 0; j < size; j++)
//	{
//		for (i = 0; i < size; i++)
//		{
//			if (i <= j) {
//				printf("%2.2f\t", mat[i +j*size]);
//			}
//			else {
//				printf(".\t");
//			}
//		}
//		printf("\n");
//	}
//


	dw_cholesky(mat, size, size, nblocks);

//	printf("Results :\n");
//
//	for (j = 0; j < size; j++)
//	{
//		for (i = 0; i < size; i++)
//		{
//			if (i <= j) {
//				printf("%2.2f\t", mat[i +j*size]);
//			}
//			else {
//				printf(".\t");
//				mat[i+j*size] = 0.0f; // debug
//			}
//		}
//		printf("\n");
//	}
//
//	printf("test results ...\n");
//	float *test_mat = malloc(size*size*sizeof(float));
//	cblas_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans, size, size, 1.0f, 
//				mat, size, 0.0f, test_mat, size);
//
//	for (j = 0; j < size; j++)
//	{
//		for (i = 0; i < size; i++)
//		{
//			if (i <= j) {
//				printf("%2.2f\t", test_mat[i +j*size]);
//			}
//			else {
//				printf(".\t");
//			}
//		}
//		printf("\n");
//	}
//
//

	return 0;
}
