#ifndef __LU_KERNELS_MODEL_H__
#define __LU_KERNELS_MODEL_H__

#include <common/util.h>

#include <datawizard/coherency.h>
#include <datawizard/interfaces/data_interface.h>

double task_11_cost(buffer_descr *descr);
double task_12_cost(buffer_descr *descr);
double task_21_cost(buffer_descr *descr);
double task_22_cost(buffer_descr *descr);

#endif // __LU_KERNELS_MODEL_H__
