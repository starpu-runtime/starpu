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

#ifndef __STARPU_MPI_FXT_H__
#define __STARPU_MPI_FXT_H__

#include <starpu.h>
#include <common/config.h>
#include <common/fxt.h>

#define FUT_MPI_BARRIER		0x5201

#ifdef USE_FXT
#define TRACE_MPI_BARRIER(rank, worldsize)	\
	FUT_DO_PROBE3(FUT_MPI_BARRIER, rank, worldsize, syscall(SYS_gettid));
#else
#define TRACE_MPI_BARRIER(a, b)		do {} while(0);
#endif



#endif // __STARPU_MPI_FXT_H__
