/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#ifndef __STARPU_BOUND_H__
#define __STARPU_BOUND_H__

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
   @defgroup API_Theoretical_Lower_Bound_on_Execution_Time Theoretical Lower Bound on Execution Time
   @brief Compute theoretical upper computation efficiency bound corresponding to some actual execution.
   @{
*/

/**
   Start recording tasks (resets stats). \p deps tells whether
   dependencies should be recorded too (this is quite expensive)

   See \ref TheoreticalLowerBoundOnExecutionTime for more details.
*/
void starpu_bound_start(int deps, int prio);

/**
   Stop recording tasks

   See \ref TheoreticalLowerBoundOnExecutionTime for more details.
*/
void starpu_bound_stop(void);

/**
   Emit the DAG that was recorded on \p output.

   See \ref TheoreticalLowerBoundOnExecutionTime for more details.
*/
void starpu_bound_print_dot(FILE *output);

/**
   Get theoretical upper bound (in ms) (needs glpk support detected by
   configure script). It returns 0 if some performance models are not
   calibrated. \p integer permits to choose between integer solving
   (which takes a long time but is correct), and relaxed solving
   (which provides an approximate solution).

   See \ref TheoreticalLowerBoundOnExecutionTime for more details.
*/
void starpu_bound_compute(double *res, double *integer_res, int integer);

/**
   Emit the Linear Programming system on \p output for the recorded
   tasks, in the lp format

   See \ref TheoreticalLowerBoundOnExecutionTime for more details.
*/
void starpu_bound_print_lp(FILE *output);

/**
   Emit the Linear Programming system on \p output for the recorded
   tasks, in the mps format

   See \ref TheoreticalLowerBoundOnExecutionTime for more details.
*/
void starpu_bound_print_mps(FILE *output);

/**
   Emit on \p output the statistics of actual execution vs theoretical
   upper bound. \p integer permits to choose between integer solving
   (which takes a long time but is correct), and relaxed solving
   (which provides an approximate solution).

   See \ref TheoreticalLowerBoundOnExecutionTime for more details.
*/
void starpu_bound_print(FILE *output, int integer);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_BOUND_H__ */
