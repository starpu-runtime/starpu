#ifndef __STRASSEN_MODELS_H__
#define __STRASSEN_MODELS_H__

#include <common/util.h>

#include <datawizard/coherency.h>
#include <datawizard/interfaces/data_interface.h>

double self_add_sub_cost(buffer_descr *descr);
double add_sub_cost(buffer_descr *descr);
double mult_cost(buffer_descr *descr);

double cuda_self_add_sub_cost(buffer_descr *descr);
double cuda_add_sub_cost(buffer_descr *descr);
double cuda_mult_cost(buffer_descr *descr);


#endif // __STRASSEN_MODELS_H__
