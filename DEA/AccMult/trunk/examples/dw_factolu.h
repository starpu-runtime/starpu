#ifndef __DW_FACTO_LU_H__
#define __DW_FACTO_LU_H__

#include <semaphore.h>

#include <cblas.h>

#include <datawizard/coherency.h>
#include <datawizard/hierarchy.h>
#include <datawizard/filters.h>

typedef struct {
	data_state *dataA;
	unsigned i;
	unsigned j;
	unsigned k;
	unsigned nblocks;
	unsigned *remaining;
	sem_t *sem;
} cl_args;


void dw_callback_codelet_update_u11(void *);
void dw_callback_codelet_update_u12_21(void *);
void dw_callback_codelet_update_u22(void *);

void dw_core_codelet_update_u11(void *);
void dw_core_codelet_update_u12(void *);
void dw_core_codelet_update_u21(void *);
void dw_core_codelet_update_u22(void *);

#endif
