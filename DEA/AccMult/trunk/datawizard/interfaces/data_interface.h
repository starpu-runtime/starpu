#ifndef __DATA_INTERFACE_H__
#define __DATA_INTERFACE_H__

#include <datawizard/data_parameters.h>
#include "blas_interface.h"
#include "vector_interface.h"
#include "csr_interface.h"
#include "csc_interface.h"
#include "bcsr_interface.h"

typedef union {
	blas_interface_t blas;	/* dense BLAS representation */
	vector_interface_t vector; /* continuous vector */
	csr_interface_t csr;	/* compressed sparse row */
	csc_interface_t csc; 	/* compressed sparse column */
	bcsr_interface_t bcsr;	/* blocked compressed sparse row */
} data_interface_t;

#endif // __DATA_INTERFACE_H__
