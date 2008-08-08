#include <common/util.h>

static uint64_t current_tag = 0;

static uint64_t get_tag()
{
	uint64_t tag = ATOMIC_ADD(&current_tag, 1);
	return tag;
}


/* dep_tag depends on that set of codelets  */

typedef enum {
	ADD,
	SUB,
	MULT,
	NONE
} operation;

static void compute_add_sub_op(data_state *A1, operation op, data_state *A2, data_state *C)
{
	/* performs C = (A op B) */

	

}

#define THRESHHOLD	1024

static void partition_matrices(data_state *A, data_state *B, data_state *C)
{
	filter f;
	f.filter_func = block_filter_func;
	f.filter_arg = 2;

	filter f2;
	f2.filter_func = vertical_block_filter_func;
	f2.filter_arg = 2;

	map_filters(A, 2, &f, &f2);
	map_filters(B, 2, &f, &f2);
	map_filters(C, 2, &f, &f2);
}

static data_state *create_tmp_matrix(data_state *M)
{
	float *data;
	data_state *state = malloc(sizeof(data_state));

	/* create a matrix with the same dimensions as M */
	uint32_t nx = get_blas_nx(M);
	uint32_t ny = get_blas_nx(M);

	ASSERT(state);

	data = malloc(nx*ny*sizeof(float));
	ASSERT(data);

	monitor_blas_data(state, 0, (uintptr_t)data, nx, nx, ny, sizeof(float));
	
	return state;
}

static void strassen(data_state *A, tag_t tagA, data_state *B, tag_t tagB, data_state *C)
{
	/* C = A * B */
	unsigned nxA = get_blas_nx(A);
	unsigned nyA = get_blas_ny(A);
	unsigned nxB = get_blas_nx(B);
	unsigned nyB = get_blas_ny(B);

	if ( (MAX(nxA,nyA) <= THRESHHOLD) && (MAX(nxB,nyB) <= THRESHHOLD))
	{
		/* don't use Strassen but a simple sequential multiplication
		 * provided this is small enough */
		compute_add_sub_op(A, MULT, B, C);
	}
	else {
		data_state *E1, *E2, *E3, *E4, *E5, *E6, *E7;
		data_state *E11, *E12, *E21, *E22, *E31, *E32, *E41, *E52, *E62, *E71;

		/* we do another level of recursion into the
		 * Strassen algorithm */
		partition_matrices(A, B, C);

		data_state *A11 = get_sub_data(&A, 1, 1);
		data_state *A12 = get_sub_data(&A, 2, 1);
		data_state *A21 = get_sub_data(&A, 1, 2);
		data_state *A22 = get_sub_data(&A, 2, 2);

		data_state *B11 = get_sub_data(&B, 1, 1);
		data_state *B12 = get_sub_data(&B, 2, 1);
		data_state *B21 = get_sub_data(&B, 1, 2);
		data_state *B22 = get_sub_data(&B, 2, 2);

		data_state *C11 = get_sub_data(&C, 1, 1);
		data_state *C12 = get_sub_data(&C, 2, 1);
		data_state *C21 = get_sub_data(&C, 1, 2);
		data_state *C22 = get_sub_data(&C, 2, 2);

		/* XXX test if matrices have the same size */


		/* E1 = (A11 - A22)x(B21 + B22) */
		E11 = create_tmp_matrix(A11);
		E12 = create_tmp_matrix(B21);
		E1  = create_tmp_matrix(C11);

		compute_add_sub_op(A11, SUB, A22, E11);
		compute_add_sub_op(B21, ADD, B22, E12);

		/* E2 = (A11 + A22)x(B11 + B22) */
		E21 = create_tmp_matrix(A11);
		E22 = create_tmp_matrix(B11);
		E2  = create_tmp_matrix(C11);

		compute_add_sub_op(A11, ADD, A22, E21);
		compute_add_sub_op(B11, ADD, B22, E22);

		/* E3 = (A11 - A21)x(B11 + B12) */
		E31 = create_tmp_matrix(A11);
		E32 = create_tmp_matrix(B11);
		E3  = create_tmp_matrix(C22);

		compute_add_sub_op(A11, SUB, A21, E31);
		compute_add_sub_op(B11, ADD, B12, E32);

		/* E4 = (A11 + A12)x(B22) */
		E41 = create_tmp_matrix(A11);
		E4  = create_tmp_matrix(C11);

		compute_add_sub_op(A11, ADD, A12, E41);

		/* E5 = (A11)x(B12 - B22) */
		E52 = create_tmp_matrix(B12);
		E5  = create_tmp_matrix(C22);

		compute_add_sub_op(B12, SUB, B22, E52);

		/* E6 = (A22)x(B21 - B11) */
		E62 = create_tmp_matrix(B21);
		E6  = create_tmp_matrix(C21);

		compute_add_sub_op(B21, SUB, B11, E62);

		/* E7 = (A21 + A22)x(B11) */
		E71 = create_tmp_matrix(A21);
		E7  = create_tmp_matrix(C22);

		compute_add_sub_op(A21, ADD, A22, E71);

		strassen(E11, E12, E1);
		strassen(E21, E22, E2);
		strassen(E31, E32, E3);
		strassen(E41, B22, E4);
		strassen(A11, E52, E5);
		strassen(A22, E62, E6);
		strassen(E71, B11, E7);

/*
		//C11 = E1 + E2 - E4 + E6;
		(C11 = E1 at that step)
		C11 += E2
		C11 += -E4 
		C11 += E6
		
		//C12 = E4 + E5;
		(C12 = E4)
		C12 += E5

		//C21 = E6 + E7;
		(C21 = E6)
		C21 += E7

		//C22 = E2 - E3 + E5 - E7;
		(C22 = E2)
		C22 += -E3
		C22 += E5
		C22 += -E7
*/


	}
}
