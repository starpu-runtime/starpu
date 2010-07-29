/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2009 (see AUTHORS file)
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

#ifndef __DW_WRITE_BACK_H__
#define __DW_WRITE_BACK_H__

#include <starpu.h>
#include <datawizard/coherency.h>

void _starpu_write_through_data(starpu_data_handle handle, uint32_t requesting_node, 
					   uint32_t write_through_mask);

#endif // __DW_WRITE_BACK_H__
