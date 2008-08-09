#ifndef __STRASSEN_H__
#define __STRASSEN_H__

#include <common/util.h>

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
	unsigned Ei12[6]; // Ei12[k] is 0, 1 or 2 (2 = finished Ei1 and Ei2)
	unsigned Ei[6];
	unsigned Cij[4];

	data_state *A, *B, *C;
	data_state *A11, *A12, *A21, *A22;
	data_state *B11, *B12, *B21, *B22;
	data_state *C11, *C12, *C21, *C22;

	data_state *E1, *E2, *E3, *E4, *E5, *E6, *E7;
	data_state *E11, *E12, *E21, *E22, *E31, *E32, *E41, *E52, *E62, *E71;

	data_state *E42, *E51, *E61, *E72;

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

#endif // __STRASSEN_H__
