#include "strassen.h"
#include "strassen_models.h"

static job_t create_job(void)
{
	codelet *cl = malloc(sizeof(codelet));
	cl->cl_arg = NULL;

	job_t j = job_create();
		j->where = CORE|CUBLAS;
		j->cl = cl;

	return j;
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

static void free_tmp_matrix(data_state *matrix)
{
	delete_data(matrix);
//	free(matrix);
}


static void partition_matrices(strassen_iter_state_t *iter)
{

	data_state *A = iter->A;
	data_state *B = iter->B;
	data_state *C = iter->C;

	filter f;
	f.filter_func = block_filter_func;
	f.filter_arg = 2;

	filter f2;
	f2.filter_func = vertical_block_filter_func;
	f2.filter_arg = 2;

	map_filters(A, 2, &f, &f2);
	map_filters(B, 2, &f, &f2);
	map_filters(C, 2, &f, &f2);

	iter->A11 = get_sub_data(A, 2, 0, 0);
	iter->A12 = get_sub_data(A, 2, 1, 0);
	iter->A21 = get_sub_data(A, 2, 0, 1);
	iter->A22 = get_sub_data(A, 2, 1, 1);

	iter->B11 = get_sub_data(B, 2, 0, 0);
	iter->B12 = get_sub_data(B, 2, 1, 0);
	iter->B21 = get_sub_data(B, 2, 0, 1);
	iter->B22 = get_sub_data(B, 2, 1, 1);

	iter->C11 = get_sub_data(C, 2, 0, 0);
	iter->C12 = get_sub_data(C, 2, 1, 0);
	iter->C21 = get_sub_data(C, 2, 0, 1);
	iter->C22 = get_sub_data(C, 2, 1, 1);

	/* TODO check that all sub-matrices have the same size */
}

static void unpartition_matrices(strassen_iter_state_t *iter)
{
	/* TODO there is no  need to actually gather those results ... */
	unpartition_data(iter->A, 0);
	unpartition_data(iter->B, 0);
	unpartition_data(iter->C, 0);
}



static void compute_add_sub_op(data_state *A1, operation op, data_state *A2, data_state *C, void (*callback)(void *), void *argcallback)
{

	/* performs C = (A op B) */
	job_t job = create_job();

	//printf("A %p B %p C %p, callback %p arg %p op %d job %p\n", A1, A2, C, callback, argcallback, op, job);

	job->nbuffers = 3;
		job->buffers[0].state = C;
		job->buffers[0].mode = W;
		job->buffers[1].state = A1;
		job->buffers[1].mode = R;
		job->buffers[2].state = A2;
		job->buffers[2].mode = R;
	
	job->cb = callback;
	job->argcb = argcallback;

	switch (op) {
		case ADD:
			ASSERT(A1);
			ASSERT(A2);
			ASSERT(C);
			job->model = &strassen_model_add_sub;
			job->cl->core_func = add_core_codelet;
			#if defined (USE_CUBLAS) || defined (USE_CUDA)
			job->cl->cublas_func = add_cublas_codelet;
			#endif
			break;
		case SUB:
			ASSERT(A1);
			ASSERT(A2);
			ASSERT(C);
			job->model = &strassen_model_add_sub;
			job->cl->core_func = sub_core_codelet;
			#if defined (USE_CUBLAS) || defined (USE_CUDA)
			job->cl->cublas_func = sub_cublas_codelet;
			#endif
			break;
		case MULT:
			ASSERT(A1);
			ASSERT(A2);
			ASSERT(C);
			job->model = &strassen_model_mult;
			job->cl->core_func = mult_core_codelet;
			#if defined (USE_CUBLAS) || defined (USE_CUDA)
			job->cl->cublas_func = mult_cublas_codelet;
			#endif
			break;
		case SELFADD:
			job->nbuffers = 2;
			job->buffers[0].mode = RW;
			job->cl->core_func = self_add_core_codelet;
			job->model = &strassen_model_self_add_sub;
			#if defined (USE_CUBLAS) || defined (USE_CUDA)
			job->cl->cublas_func = self_add_cublas_codelet;
			#endif
			break;
		case SELFSUB:
			job->nbuffers = 2;
			job->buffers[0].mode = RW;
			job->cl->core_func = self_sub_core_codelet;
			job->model = &strassen_model_self_add_sub;
			#if defined (USE_CUBLAS) || defined (USE_CUDA)
			job->cl->cublas_func = self_sub_cublas_codelet;
			#endif
			break;
		default:
			ASSERT(0);
	}

	push_task(job);
	// XXX this could made out of various codelets, for now, a single one
}

/* Cij +=/-= Ek is done */
void phase_3_callback_function(void *_arg)
{
	unsigned cnt, use_cnt;
	phase3_t *arg = _arg;

	unsigned i = arg->i;
	strassen_iter_state_t *iter = arg->iter;

	free(arg);

	use_cnt = ATOMIC_ADD(&iter->Ei_remaining_use[i], -1);
	if (use_cnt == 0) 
	{
		/* no one needs Ei anymore : free it */
		switch (i) {
			case 0:
				free_tmp_matrix(iter->E1);
				break;
			case 1:
				free_tmp_matrix(iter->E2);
				break;
			case 2:
				free_tmp_matrix(iter->E3);
				break;
			case 3:
				free_tmp_matrix(iter->E4);
				break;
			case 4:
				free_tmp_matrix(iter->E5);
				break;
			case 5:
				free_tmp_matrix(iter->E6);
				break;
			case 6:
				free_tmp_matrix(iter->E7);
				break;
			default:
				ASSERT(0);
		}
	}

	cnt = ATOMIC_ADD(&iter->counter, -1);
	if (cnt == 0)
	{
		/* the entire strassen iteration is done ! */
		unpartition_matrices(iter);

		// XXX free the Ei
		ASSERT(iter->strassen_iter_callback);
		iter->strassen_iter_callback(iter->argcb);

		free(iter);
	}
}



/* Ei is computed */
void phase_2_callback_function(void *_arg)
{
	phase2_t *arg = _arg;

	strassen_iter_state_t *iter = arg->iter;
	unsigned i = arg->i;

	free(arg);

	phase3_t *arg1, *arg2;
	arg1 = malloc(sizeof(phase3_t));
	arg2 = malloc(sizeof(phase3_t));

	arg1->iter = iter;
	arg2->iter = iter;

	arg1->i = i;
	arg2->i = i;

	switch (i) {
		case 0:
			free(arg2); // will not be needed .. 
			free_tmp_matrix(iter->E11);
			free_tmp_matrix(iter->E12);
			/* C11 += E1 */
			compute_add_sub_op(iter->E1, SELFADD, NULL, iter->C11, phase_3_callback_function, arg1);
			break;
		case 1:
			free_tmp_matrix(iter->E21);
			free_tmp_matrix(iter->E22);
			/* C11 += E2 */
			compute_add_sub_op(iter->E2, SELFADD, NULL, iter->C11, phase_3_callback_function, arg1);
			/* C22 += E2 */
			compute_add_sub_op(iter->E2, SELFADD, NULL, iter->C22, phase_3_callback_function, arg2);
			break;
		case 2:
			free(arg2); // will not be needed .. 
			free_tmp_matrix(iter->E31);
			free_tmp_matrix(iter->E32);
			/* C22 -= E3 */
			compute_add_sub_op(iter->E3, SELFSUB, NULL, iter->C22, phase_3_callback_function, arg1);
			break;
		case 3:
			free_tmp_matrix(iter->E41);
			/* C11 -= E4 */
			compute_add_sub_op(iter->E4, SELFSUB, NULL, iter->C11, phase_3_callback_function, arg1);
			/* C12 += E4 */
			compute_add_sub_op(iter->E4, SELFADD, NULL, iter->C12, phase_3_callback_function, arg2);
			break;
		case 4:
			free_tmp_matrix(iter->E52);
			/* C12 += E5 */
			compute_add_sub_op(iter->E5, SELFADD, NULL, iter->C12, phase_3_callback_function, arg1);
			/* C22 += E5 */
			compute_add_sub_op(iter->E5, SELFADD, NULL, iter->C22, phase_3_callback_function, arg2);
			break;
		case 5:
			free_tmp_matrix(iter->E62);
			/* C11 += E6 */
			compute_add_sub_op(iter->E6, SELFADD, NULL, iter->C11, phase_3_callback_function, arg1);
			/* C21 += E6 */
			compute_add_sub_op(iter->E6, SELFADD, NULL, iter->C21, phase_3_callback_function, arg2);
			break;
		case 6:
			free_tmp_matrix(iter->E71);
			/* C21 += E7 */
			compute_add_sub_op(iter->E7, SELFADD, NULL, iter->C21, phase_3_callback_function, arg1);
			/* C22 -= E7 */
			compute_add_sub_op(iter->E7, SELFSUB, NULL, iter->C22, phase_3_callback_function, arg2);
			break;
		default:
			ASSERT(0);
	}
}


/* computes Ei */
static void _strassen_phase_2(strassen_iter_state_t *iter, unsigned i)
{
	phase2_t *phase_2_arg = malloc(sizeof(phase2_t));

	phase_2_arg->iter = iter;
	phase_2_arg->i = i;

	/* XXX */
	data_state *A;
	data_state *B;
	data_state *C;

	switch (i) {
		case 0:
			A = iter->E11; B = iter->E12;
			iter->E1 = create_tmp_matrix(A);
			C = iter->E1;
			break;
		case 1:
			A = iter->E21; B = iter->E22;
			iter->E2 = create_tmp_matrix(A);
			C = iter->E2;
			break;
		case 2:
			A = iter->E31; B = iter->E32;
			iter->E3 = create_tmp_matrix(A);
			C = iter->E3;
			break;
		case 3:
			A = iter->E41; B = iter->E42;
			iter->E4 = create_tmp_matrix(A);
			C = iter->E4;
			break;
		case 4:
			A = iter->E51; B = iter->E52;
			iter->E5 = create_tmp_matrix(A);
			C = iter->E5;
			break;
		case 5:
			A = iter->E61; B = iter->E62;
			iter->E6 = create_tmp_matrix(A);
			C = iter->E6;
			break;
		case 6:
			A = iter->E71; B = iter->E72;
			iter->E7 = create_tmp_matrix(A);
			C = iter->E7;
			break;
		default:
			ASSERT(0);
	}

	ASSERT(A);
	ASSERT(B);
	ASSERT(C);

	// DEBUG XXX
	//compute_add_sub_op(A, MULT, B, C, phase_2_callback_function, phase_2_arg);
	strassen(A, B, C, phase_2_callback_function, phase_2_arg, iter->reclevel-1);
}


#define THRESHHOLD	128

static void phase_1_callback_function(void *_arg)
{

	phase1_t *arg = _arg;
	strassen_iter_state_t *iter = arg->iter;
	unsigned i = arg->i;

	free(arg);

	unsigned cnt = ATOMIC_ADD(&iter->Ei12[i], +1);

	if (cnt == 2) {
		/* Ei1 and Ei2 are ready, compute Ei */
		_strassen_phase_2(iter, i);
	}
}

/* computes Ei1 or Ei2 with i in 0-6 */
static void _strassen_phase_1(data_state *A1, operation opA, data_state *A2,
			      data_state *C, strassen_iter_state_t *iter, unsigned i)
{
	phase1_t *phase_1_arg = malloc(sizeof(phase1_t));
	phase_1_arg->iter = iter;
	phase_1_arg->i = i;

	compute_add_sub_op(A1, opA, A2, C, phase_1_callback_function, phase_1_arg);
}

strassen_iter_state_t *init_strassen_iter_state(data_state *A, data_state *B, data_state *C, void (*strassen_iter_callback)(void *), void *argcb)
{
	strassen_iter_state_t *iter_state = malloc(sizeof(strassen_iter_state_t));

	iter_state->Ei12[0] = 0;
	iter_state->Ei12[1] = 0;
	iter_state->Ei12[2] = 0;
	iter_state->Ei12[3] = 1; // E42 = B22
	iter_state->Ei12[4] = 1; // E51 = A11
	iter_state->Ei12[5] = 1; // E61 = A22
	iter_state->Ei12[6] = 1; // E72 = B11

	iter_state->Ei_remaining_use[0] = 1; 
	iter_state->Ei_remaining_use[1] = 2;
	iter_state->Ei_remaining_use[2] = 1;
	iter_state->Ei_remaining_use[3] = 2;
	iter_state->Ei_remaining_use[4] = 2;
	iter_state->Ei_remaining_use[5] = 2;
	iter_state->Ei_remaining_use[6] = 2;

	unsigned i;
	for (i = 0; i < 6; i++)
	{
		iter_state->Ei[i] = 0;
	}

	for (i = 0; i < 4; i++)
	{
		iter_state->Cij[i] = 0;
	}

	iter_state->strassen_iter_callback = strassen_iter_callback;
	iter_state->argcb = argcb;

	iter_state->A = A;
	iter_state->B = B;
	iter_state->C = C;

	iter_state->counter = 12;

	return iter_state;
}

static void _do_strassen(data_state *A, data_state *B, data_state *C, void (*strassen_iter_callback)(void *), void *argcb, unsigned reclevel)
{
	/* do one level of recursion in the strassen algorithm */
	strassen_iter_state_t *iter = init_strassen_iter_state(A, B, C, strassen_iter_callback, argcb);

	partition_matrices(iter);
	iter->reclevel = reclevel;

	/* some Eij are already known */
	iter->E11 = create_tmp_matrix(iter->A11);
	iter->E12 = create_tmp_matrix(iter->B21);
	iter->E21 = create_tmp_matrix(iter->A11);
	iter->E22 = create_tmp_matrix(iter->B11);
	iter->E31 = create_tmp_matrix(iter->A11);
	iter->E32 = create_tmp_matrix(iter->B11);
	iter->E41 = create_tmp_matrix(iter->A11);
	iter->E42 = iter->B22;
	iter->E51 = iter->A11;
	iter->E52 = create_tmp_matrix(iter->B12);
	iter->E61 = iter->A22;
	iter->E62 = create_tmp_matrix(iter->B21);
	iter->E71 = create_tmp_matrix(iter->A21);
	iter->E72 = iter->B11;

	/* compute all Eij */
	_strassen_phase_1(iter->A11, SUB, iter->A22, iter->E11, iter, 0);
	_strassen_phase_1(iter->B21, ADD, iter->B22, iter->E12, iter, 0);
	_strassen_phase_1(iter->A11, ADD, iter->A22, iter->E21, iter, 1);
	_strassen_phase_1(iter->B11, ADD, iter->B22, iter->E22, iter, 1);
	_strassen_phase_1(iter->A11, SUB, iter->A21, iter->E31, iter, 2);
	_strassen_phase_1(iter->B11, ADD, iter->B12, iter->E32, iter, 2);
	_strassen_phase_1(iter->A11, ADD, iter->A12, iter->E41, iter, 3);
	_strassen_phase_1(iter->B12, SUB, iter->B22, iter->E52, iter, 4);
	_strassen_phase_1(iter->B21, SUB, iter->B11, iter->E62, iter, 5);
	_strassen_phase_1(iter->A21, ADD, iter->A22, iter->E71, iter, 6);
}


void strassen(data_state *A, data_state *B, data_state *C, void (*callback)(void *), void *argcb, unsigned reclevel)
{
	/* C = A * B */
	if ( reclevel == 0 )
	{
		/* don't use Strassen but a simple sequential multiplication
		 * provided this is small enough */
		compute_add_sub_op(A, MULT, B, C, callback, argcb);
	}
	else {
		_do_strassen(A, B, C, callback, argcb, reclevel);
	}
}
