#ifndef __DATA_INTERFACE_H__
#define __DATA_INTERFACE_H__

#include <stdio.h>

#include <datawizard/data_parameters.h>
#include "blas_interface.h"
#include "vector_interface.h"
#include "csr_interface.h"
#include "csc_interface.h"
#include "bcsr_interface.h"

#ifdef USE_GORDON
/* to get the gordon_strideSize_t data structure from gordon */
#include <cell/gordon/gordon.h>
#endif

typedef union {
	blas_interface_t blas;	/* dense BLAS representation */
	vector_interface_t vector; /* continuous vector */
	csr_interface_t csr;	/* compressed sparse row */
	csc_interface_t csc; 	/* compressed sparse column */
	bcsr_interface_t bcsr;	/* blocked compressed sparse row */
} data_interface_t;

struct data_interface_ops_t {
	size_t (*allocate_data_on_node)(struct data_state_t *state,
					uint32_t node);
	void (*liberate_data_on_node)(data_interface_t *interface,
					uint32_t node);
	int (*copy_data_1_to_1)(struct data_state_t *state, 
					uint32_t src, uint32_t dst);
	size_t (*dump_data_interface)(data_interface_t *interface, 
					void *buffer);
	size_t (*get_size)(struct data_state_t *state);
	uint32_t (*footprint)(struct data_state_t *state, uint32_t hstate);
	void (*display)(struct data_state_t *state, FILE *f);
#ifdef USE_GORDON
	int (*convert_to_gordon)(data_interface_t *interface, uint64_t *ptr, gordon_strideSize_t *ss); 
#endif
	/* an identifier that is unique to each interface */
	uint32_t interfaceid;
};

#endif // __DATA_INTERFACE_H__
