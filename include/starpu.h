#ifndef __STARPU_H__
#define __STARPU_H__

#include <stdlib.h>
#include <stdint.h>

#include <starpu-data.h>
#include <starpu-perfmodel.h>
#include <starpu-task.h>

void starpu_init(void);
void starpu_shutdown(void);

#endif // __STARPU_H__
