/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011  Université de Bordeaux 1
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef _STARPU_PARAMETERS_H
#define _STARPU_PARAMETERS_H

/* Parameters which are not worth being added to ./configure options, but
 * still interesting to easily change */

/* The dmda scheduling policy uses
 *
 * alpha * T_computation + beta * T_communication + gamma * Consumption
 *
 * Here are the default values of alpha, beta, gamma
 */

#define STARPU_DEFAULT_ALPHA 1.0
#define STARPU_DEFAULT_BETA 1.0
#define STARPU_DEFAULT_GAMMA 1000.0

#endif /* _STARPU_PARAMETERS_H */
