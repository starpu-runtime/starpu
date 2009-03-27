#ifndef __STARPU_H__
#define __STARPU_H__

#include <stdlib.h>
#include <stdint.h>

#include <starpu_config.h>
#include <starpu-data.h>
#include <starpu-perfmodel.h>
#include <starpu-task.h>

/* Initialization method: it must be called prior to any other StarPU call */
void starpu_init(void);

/* Shutdown method: note that statistics are only generated once StarPU is
 * shutdown */
void starpu_shutdown(void);

#endif // __STARPU_H__
