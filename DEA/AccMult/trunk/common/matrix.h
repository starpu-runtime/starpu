#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <core/jobs.h>
#include "util.h"

void matrix_fill_rand(matrix *);
void matrix_fill_zero(matrix *);
void alloc_matrix(matrix *, unsigned, unsigned);
void free_matrix(matrix *);
void display_matrix(matrix *);
void display_submatrix(submatrix *);
void compare_matrix(matrix *, matrix *, float);

#endif // __MATRIX_H__
