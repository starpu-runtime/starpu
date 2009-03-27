#ifndef __STARPU_H__
#define __STARPU_H__

#include <stdlib.h>
#include <stdint.h>

#include <starpu-data.h>
#include <starpu-perfmodel.h>
#include <starpu-task.h>

void init_machine(void);
void terminate_machine(void);

#endif // __STARPU_H__
