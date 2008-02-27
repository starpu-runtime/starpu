#ifndef __COMP_H__
#define __COMP_H__

#include "jobs.h"

void mult(matrix *, matrix *, matrix *);
void dummy_mult(submatrix *, submatrix *, submatrix *);
void ref_mult(matrix *, matrix *, matrix *);
void cblas_mult(submatrix *, submatrix *, submatrix *);

#endif // __COMP_H__
