#include <string.h>
#include <stdint.h>
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
	uint32_t *colind;
	uint32_t *rowptr;
} bcsr_t;


/* directly read input from a file */
bcsr_t *mm_file_to_bcsr(char *filename, unsigned c, unsigned r);

/* read the matrix as a set of valuated coordinates */
bcsr_t *mm_to_bcsr(unsigned nz, unsigned *I, unsigned *J, float *val, unsigned c, unsigned r);
