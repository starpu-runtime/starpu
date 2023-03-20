/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_MAX_FPGA_H__
#define __STARPU_MAX_FPGA_H__

#include <starpu_config.h>

#if defined STARPU_USE_MAX_FPGA
#include <MaxSLiCInterface.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
   @defgroup API_Max_FPGA_Extensions Maxeler FPGA Extensions
   @{
*/

/**
   This specifies a Maxeler file to be loaded on some engines.
 */
struct starpu_max_load
{
	max_file_t *file;	       /**< Provide the file to be loaded */
	const char *engine_id_pattern; /**< Provide the engine(s) on which to be loaded, following
					     the Maxeler engine naming, i.e. typically
                                             "*:0", "*:1", etc.
                                             In an array of struct starpu_max_load, only one can have
                                             the "*" specification.  */
};

/**
   Maxeler engine of the current worker.
   See \ref MaxFPGAExample for more details.
 */
max_engine_t *starpu_max_fpga_get_local_engine(void);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* STARPU_USE_MAX_FPGA */
#endif /* __STARPU_MAX_FPGA_H__ */
