#ifndef __DATA_INTERFACE_H__
#define __DATA_INTERFACE_H__

#include <datawizard/data_parameters.h>
#include "blas_interface.h"
#include "csr_interface.h"
#include "csc_interface.h"

typedef union {
	blas_interface_t blas;	/* dense BLAS representation */
	csr_interface_t csr;	/* compressed sparse row */
	csc_interface_t csc; 	/* compressed sparse column */
} data_interface_t;

#endif // __DATA_INTERFACE_H__
