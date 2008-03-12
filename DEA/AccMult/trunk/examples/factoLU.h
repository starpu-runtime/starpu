#ifndef __FACTO_LU_H__
#define __FACTO_LU_H__

#include <semaphore.h>

typedef struct subproblem_t {
	submatrix *LU;
	submatrix *LU11;
	submatrix *LU12;
	submatrix *LU21;
	submatrix *LU22;
	sem_t *sem; /* for the entire subproblem */
	unsigned at_counter_lu12_21;
	unsigned at_counter_lu22;
	unsigned grain;
	unsigned rec_level; // for debug ... 
} subproblem;

typedef struct u11_args_t {
	subproblem *subp;
} u11_args;

typedef struct u1221_args_t {
	subproblem *subp;
	unsigned xa;
	unsigned xb;
	unsigned ya;
	unsigned yb;
} u1221_args;

typedef struct u22_args_t {
	subproblem *subp;
	unsigned xa;
	unsigned xb;
	unsigned ya;
	unsigned yb;
} u22_args;

void callback_codelet_update_u11(void *);
void callback_codelet_update_u12_21(void *);
void callback_codelet_update_u22(void *);

void core_codelet_update_u11(void *);
void core_codelet_update_u12(void *);
void core_codelet_update_u21(void *);
void core_codelet_update_u22(void *);

#endif
