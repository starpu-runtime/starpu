/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#ifndef __FUNC_SGEMM_IBM_H__
#define __FUNC_SGEMM_IBM_H__

#include <stdint.h>

struct ibm_sgemm_block_conf {
	uint32_t m;
	uint32_t n;
	uint32_t k;
	uint32_t pad;
};

#endif // __FUNC_SGEMM_IBM_H__
