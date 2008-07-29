#ifndef __BLAS_MODEL_H__
#define __BLAS_MODEL_H__

#include <common/util.h>

#include <datawizard/coherency.h>
#include <datawizard/interfaces/data_interface.h>

double gemm_cost(buffer_descr *descr);

#endif // __BLAS_MODEL_H__
