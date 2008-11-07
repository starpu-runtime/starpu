#ifndef __STRASSEN_H__
#define __STRASSEN_H__

#include <semaphore.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <core/dependencies/tags.h>
#include <common/timing.h>
#include <common/util.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <pthread.h>
#include <signal.h>
#include <cblas.h>
#include <common/timing.h>

#include <datawizard/datawizard.h>

#include <task-models/blas_model.h>

#include <common/fxt.h>

#if defined (USE_CUBLAS) || defined (USE_CUDA)
#include <cuda.h>
#endif

typedef enum {
	ADD,
	SUB,
	MULT,
	SELFADD,
	SELFSUB,
	NONE
} operation;

typedef struct {
	/* monitor the progress of the algorithm */
	unsigned Ei12[7]; // Ei12[k] is 0, 1 or 2 (2 = finished Ei1 and Ei2)
	unsigned Ei[7];
	unsigned Ei_remaining_use[7];
	unsigned Cij[4];

	data_state *A, *B, *C;
	data_state *A11, *A12, *A21, *A22;
	data_state *B11, *B12, *B21, *B22;
	data_state *C11, *C12, *C21, *C22;

	data_state *E1, *E2, *E3, *E4, *E5, *E6, *E7;
	data_state *E11, *E12, *E21, *E22, *E31, *E32, *E41, *E52, *E62, *E71;

	data_state *E42, *E51, *E61, *E72;

	unsigned reclevel;
	
	/* */
	unsigned counter;

	/* called at the end of the iteration */
	void (*strassen_iter_callback)(void *);
	void *argcb;
} strassen_iter_state_t;

typedef struct {
	strassen_iter_state_t *iter;

	/* phase 1 computes Ei1 or Ei2 with i in 0-6 */
	unsigned i;
} phase1_t;

typedef struct {
	strassen_iter_state_t *iter;

	/* phase 2 computes Ei with i in 0-6 */
	unsigned i;
} phase2_t;

typedef struct {
	strassen_iter_state_t *iter;

	/* phase 2 computes Ei with i in 0-6 */
	unsigned i;
} phase3_t;

void mult_core_codelet(data_interface_t *descr, __attribute__((unused))  void *arg);
void sub_core_codelet(data_interface_t *descr, __attribute__((unused))  void *arg);
void add_core_codelet(data_interface_t *descr, __attribute__((unused))  void *arg);
void self_add_core_codelet(data_interface_t *descr, __attribute__((unused))  void *arg);
void self_sub_core_codelet(data_interface_t *descr, __attribute__((unused))  void *arg);

#if defined (USE_CUBLAS) || defined (USE_CUDA)
void mult_cublas_codelet(data_interface_t *descr, __attribute__((unused))  void *arg);
void sub_cublas_codelet(data_interface_t *descr, __attribute__((unused))  void *arg);
void add_cublas_codelet(data_interface_t *descr, __attribute__((unused))  void *arg);
void self_add_cublas_codelet(data_interface_t *descr, __attribute__((unused))  void *arg);
void self_sub_cublas_codelet(data_interface_t *descr, __attribute__((unused))  void *arg);
#endif

void strassen(data_state *A, data_state *B, data_state *C, void (*callback)(void *), void *argcb, unsigned reclevel);

extern struct perfmodel_t strassen_model_mult;
extern struct perfmodel_t strassen_model_add_sub;
extern struct perfmodel_t strassen_model_self_add_sub;

#endif // __STRASSEN_H__
