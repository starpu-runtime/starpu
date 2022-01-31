/**
 * Document: MaxCompiler Tutorial (maxcompiler-tutorial.pdf)
 * Chapter: 13      Example: 2      Name: LMem Loopback
 * MaxFile name: LMemLoopback
 * Summary:
 *        Adds two LMem input streams and writes the result to LMem.
 */

#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "LMemLoopback.h"
#include <MaxSLiCInterface.h>

int check(int size, int32_t *outData, int32_t *inA, int32_t *inB)
{
	int status = 0;
	for (int i = 0; i < size; i++)
	{
		if (outData[i] != inA[i] + inB[i])
		{
			fprintf(stderr, "[%d] Verification error, out: %u != expected: %u\n",
				i, outData[i], inA[i] + inB[i]);
			status = 1;
		}
	}
	return status;
}

int main()
{
	const int size = 384;
	int sizeBytes = size * sizeof(int32_t);
	int32_t *inA = (int32_t*) malloc(sizeBytes);
	int32_t *inB = (int32_t*) malloc(sizeBytes);

	for (int i = 0; i < size; i++)
	{
		inA[i] = i;
		inB[i] = size - i;
	}

	printf("Loading DFE memory.\n");
	LMemLoopback_writeLMem(size, 0, inA);
	LMemLoopback_writeLMem(size, size, inB);

	printf("Running DFE.\n");
	LMemLoopback(size);

	printf("Reading DFE memory.\n");
	int32_t *outData = (int32_t*) malloc(sizeBytes);
	LMemLoopback_readLMem(size, 2 * size, outData);

	int status = check(size, outData, inA, inB);
	if (status)
		printf("Test failed.\n");
	else
		printf("Test passed OK!\n");

	return status;
}
