#ifndef DGELS_H 
#define DGELS_H

#include "f2c.h"

int _starpu_dgels_(char *trans, integer *m, integer *n, integer *nrhs, doublereal *a, integer *lda, doublereal *b, integer *ldb, doublereal *work, integer *lwork, integer *info);

#endif
