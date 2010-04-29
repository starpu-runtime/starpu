/*
 * StarPU
 * Copyright (C) INRIA 2008-2010 (see AUTHORS file)
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

__kernel void incA(__global unsigned* input) 
{
	const int i = get_global_id(0);
	if (i == 0)
		input[i] ++;
}

__kernel void incC(__global unsigned* input) 
{
	const int i = get_global_id(0);
	if (i == 2)
		input[i] ++;
}

