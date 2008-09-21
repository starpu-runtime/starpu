#ifndef __DW_BLOCK_SPMV_H__
#define __DW_BLOCK_SPMV_H__

#include <semaphore.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <core/dependencies/tags.h>
#include <common/timing.h>
#include <common/util.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/types.h>
#include <pthread.h>
#include <signal.h>
#include <cblas.h>
#include <common/timing.h>

#include <datawizard/coherency.h>
#include <datawizard/hierarchy.h>
#include <datawizard/interfaces/blas_interface.h>
#include <datawizard/interfaces/csr_interface.h>
#include <datawizard/interfaces/bcsr_interface.h>
#include <datawizard/interfaces/vector_interface.h>
#include <datawizard/interfaces/blas_filters.h>
#include <datawizard/interfaces/csr_filters.h>
#include <datawizard/interfaces/bcsr_filters.h>
#include <datawizard/interfaces/vector_filters.h>

#endif // __DW_BLOCK_SPMV_H__
