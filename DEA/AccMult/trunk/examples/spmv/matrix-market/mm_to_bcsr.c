
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

void print_block(tmp_block_t *block, unsigned r, unsigned c)
{
	printf(" **** block %d %d **** \n", block->i, block->j);

	int i, j;
	for (j = 0; j < r; j++) {
		for (i = 0; i < c; i++) {
			printf("%2.2f\t", block->val[i + j*c]);
		}
		printf("\n");
	}
}

void print_all_blocks(tmp_block_t *block_list, unsigned r, unsigned c)
{
	tmp_block_t *current_block = block_list;

	while(current_block) {
		print_block(current_block, r, c);

		current_block = current_block->next;
	};

}

tmp_block_t *search_block(tmp_block_t *block_list, unsigned i, unsigned j)
{
	tmp_block_t *current_block = block_list;
	//printf("search %d %d\n", i, j);

	while (current_block) {
		if ((current_block->i == i) && (current_block->j == j)) 
		{
			/* we found the block */
			return current_block;
		}

		current_block = current_block->next;
	};

	/* no entry was found ... */
	return NULL;
}

tmp_block_t *create_block(unsigned c, unsigned r)
{
	tmp_block_t *block;

	block = malloc(sizeof(tmp_block_t));
	block->val = calloc(c*r, sizeof(float));


	return block;
}

unsigned next_block_is_bigger(tmp_block_t *block, unsigned i, unsigned j)
{
	tmp_block_t *next = block->next;

	if (next)
	{
		/* we evaluate lexical order */
		if (next->j < j)
			return 0;

		if (next->j > j)
			return 1;

		/* next->j == j */
		return (next->i > i);
	}

	/* this is the last block, so it's bigger */
	return 1;
}

void insert_block(tmp_block_t *block, tmp_block_t **block_list, unsigned i, unsigned j)
{
	///* insert block at the beginning of the list */
	//block->next = *block_list;
	//*block_list = block;

	/* insert the block in lexicographical order */
	/* first find an element that is bigger, then insert the block just before it */
	tmp_block_t *current_block = *block_list;

	if (!current_block) {
		/* list was empty */
		*block_list = block;
		block->next = NULL;
		return;
	}

	while (current_block) {
		if (next_block_is_bigger(current_block, i, j)) {
			/* insert block here */
			block->next = current_block->next;
			current_block->next = block;
			return;
		}

		current_block = current_block->next;
	};

	/* should not be reached ! */
}

void insert_elem(tmp_block_t **block_list, unsigned abs_i, unsigned abs_j, float val, unsigned c, unsigned r)
{
	/* we are looking for the block that contains (abs_i, abs_j) (abs = absolute) */
	unsigned i,j;

	i = abs_i / c;
	j = abs_j / r;

	tmp_block_t *block;

	block = search_block(*block_list, i, j);

	if (!block) {
		/* the block does not exist yet */
		/* create it */
		block = create_block(c, r);

		block->i = i;
		block->j = j;
		
		//printf("create block %d %d !\n", i, j);

		/* insert it in the block list */
		insert_block(block, block_list, i, j);
	}

	/* now insert the value in the corresponding block */
	unsigned local_i, local_j, local_index;

	local_i = abs_i % c;
	local_j = abs_j % r;
	local_index = local_j * c + local_i;
	
	block->val[local_index] = val;

	printf("prout %p\n", *block_list);
}


/* matrix market to BCSR */
tmp_block_t * mm_to_bcsr(int nz, unsigned *I, unsigned *J, float *val, unsigned c, unsigned r)
{
	int elem;

	/* at first, the list of block is empty */
	tmp_block_t *block_list = NULL;

	for (elem = 0; elem < nz; elem++)
	{
		insert_elem(&block_list, I[elem], J[elem], val[elem], c, r);
	}

	return block_list;
}


int main(int argc, char *argv[])
{
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;   
    unsigned i, *I, *J;
    float *val;

    if (argc < 2)
	{
		fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
		exit(1);
	}
    else    
    { 
        if ((f = fopen(argv[1], "r")) == NULL) 
            exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }


    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
            mm_is_sparse(matcode) )
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* find out size of sparse matrix .... */

    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
        exit(1);


    /* reseve memory for matrices */

    I = (int *) malloc(nz * sizeof(int));
    J = (int *) malloc(nz * sizeof(int));
    /* XXX float ! */
    val = (float *) malloc(nz * sizeof(float));

    for (i=0; i<nz; i++)
    {
        fscanf(f, "%d %d %f\n", &I[i], &J[i], &val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }

    if (f !=stdin) fclose(f);

    /************************/
    /* now write out matrix */
    /************************/

   // mm_write_banner(stdout, matcode);
  //  mm_write_mtx_crd_size(stdout, M, N, nz);
   // for (i=0; i<nz; i++)
   //     fprintf(stdout, "%d %d %20.19g\n", I[i]+1, J[i]+1, val[i]);

	unsigned r = 4;
	unsigned c = 6;

   tmp_block_t *block_list;
   block_list = mm_to_bcsr(nz, I, J, val, c, r);

   print_all_blocks(block_list, r, c);

	return 0;
}

