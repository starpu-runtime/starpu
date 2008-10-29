#ifndef __REGRESSION_H__
#define __REGRESSION_H__

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <common/util.h>
#include <core/perfmodel/perfmodel.h>

int regression_non_linear_power(struct history_list_t *ptr, double *a, double *b, double *c);

#endif // __REGRESSION_H__ 
