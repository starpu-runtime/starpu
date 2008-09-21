#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"

/* convert a matrix stored in a file with the matrix market format into the 
 * BCSR format */

typedef struct tmp_block {
	/* we have a linked list of blocks */
	struct tmp_block *next;

	/* column i, row j*/
	unsigned i, j;
	
	float *val;

} tmp_block_t;

typedef struct {
	unsigned r,c;
	unsigned nnz_blocks;
	unsigned nrows_blocks;

	float *val;
	unsigned *colind;
	unsigned *rowptr;
} bcsr_t;


bcsr_t *mm_file_to_bcsr(char *filename, unsigned c, unsigned r);
