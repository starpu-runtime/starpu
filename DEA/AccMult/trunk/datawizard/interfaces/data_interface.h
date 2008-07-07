#ifndef __DATA_INTERFACE_H__
#define __DATA_INTERFACE_H__

#include <datawizard/data_parameters.h>
#include "blas_interface.h"
#include "crs_interface.h"

typedef union {
	blas_interface_t blas;
	crs_interface_t crs;
} data_interface_t;

#endif // __DATA_INTERFACE_H__
