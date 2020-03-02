/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2012       Vincent Danjean
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

#include "socl.h"
#include "getinfo.h"

/**
 * \brief Get information about StarPU platform
 *
 * \param[in] platform StarPU platform ID or NULL
 */
CL_API_SUFFIX__VERSION_1_0
CL_API_ENTRY cl_int CL_API_CALL
soclGetPlatformInfo(cl_platform_id   platform,
		    cl_platform_info param_name,
		    size_t           param_value_size,
		    void *           param_value,
		    size_t *         param_value_size_ret)
{
	if (platform != NULL && platform != &socl_platform)
		return CL_INVALID_PLATFORM;

	switch (param_name)
	{
		INFO_CASE_STRING(CL_PLATFORM_PROFILE, SOCL_PROFILE);
		INFO_CASE_STRING(CL_PLATFORM_VERSION, SOCL_VERSION);
		INFO_CASE_STRING(CL_PLATFORM_NAME,    SOCL_PLATFORM_NAME);
		INFO_CASE_STRING(CL_PLATFORM_VENDOR,  SOCL_VENDOR);
		INFO_CASE_STRING(CL_PLATFORM_EXTENSIONS, SOCL_PLATFORM_EXTENSIONS);
		INFO_CASE_STRING(CL_PLATFORM_ICD_SUFFIX_KHR, SOCL_PLATFORM_ICD_SUFFIX_KHR);
	default:
		return CL_INVALID_VALUE;
	}

	return CL_SUCCESS;
}
