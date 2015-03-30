/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010  Université de Bordeaux
 * Copyright (C) 2010, 2012  CNRS
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

#ifndef __STARPU_MPI_FXT_H__
#define __STARPU_MPI_FXT_H__

#include <starpu.h>
#include <common/config.h>
#include <common/fxt.h>

#ifdef __cplusplus
extern "C" {
#endif

#define _STARPU_MPI_FUT_START				0x5201
#define _STARPU_MPI_FUT_STOP				0x5202
#define _STARPU_MPI_FUT_BARRIER				0x5203
#define _STARPU_MPI_FUT_ISEND_SUBMIT_BEGIN		0x5204
#define _STARPU_MPI_FUT_ISEND_SUBMIT_END		0x5205
#define _STARPU_MPI_FUT_IRECV_SUBMIT_BEGIN		0x5206
#define _STARPU_MPI_FUT_IRECV_SUBMIT_END		0x5207
#define _STARPU_MPI_FUT_ISEND_COMPLETE_BEGIN		0x5208
#define _STARPU_MPI_FUT_ISEND_COMPLETE_END		0x5209
#define _STARPU_MPI_FUT_IRECV_COMPLETE_BEGIN		0x5210
#define _STARPU_MPI_FUT_IRECV_COMPLETE_END		0x5211
#define _STARPU_MPI_FUT_SLEEP_BEGIN			0x5212
#define _STARPU_MPI_FUT_SLEEP_END			0x5213
#define _STARPU_MPI_FUT_DTESTING_BEGIN			0x5214
#define _STARPU_MPI_FUT_DTESTING_END			0x5215
#define _STARPU_MPI_FUT_UTESTING_BEGIN			0x5216
#define _STARPU_MPI_FUT_UTESTING_END			0x5217
#define _STARPU_MPI_FUT_UWAIT_BEGIN			0x5218
#define _STARPU_MPI_FUT_UWAIT_END			0x5219

#ifdef STARPU_USE_FXT
#define _STARPU_MPI_TRACE_START(rank, worldsize)	\
	FUT_DO_PROBE3(_STARPU_MPI_FUT_START, (rank), (worldsize), _starpu_gettid());
#define _STARPU_MPI_TRACE_STOP(rank, worldsize)	\
	FUT_DO_PROBE3(_STARPU_MPI_FUT_STOP, (rank), (worldsize), _starpu_gettid());
#define _STARPU_MPI_TRACE_BARRIER(rank, worldsize, key)	\
	FUT_DO_PROBE4(_STARPU_MPI_FUT_BARRIER, (rank), (worldsize), (key), _starpu_gettid());
#define _STARPU_MPI_TRACE_ISEND_SUBMIT_BEGIN(dest, mpi_tag, size)	\
	FUT_DO_PROBE4(_STARPU_MPI_FUT_ISEND_SUBMIT_BEGIN, (dest), (mpi_tag), (size), _starpu_gettid());
#define _STARPU_MPI_TRACE_ISEND_SUBMIT_END(dest, mpi_tag, size)	\
	FUT_DO_PROBE4(_STARPU_MPI_FUT_ISEND_SUBMIT_END, (dest), (mpi_tag), (size), _starpu_gettid());
#define _STARPU_MPI_TRACE_IRECV_SUBMIT_BEGIN(src, mpi_tag)	\
	FUT_DO_PROBE3(_STARPU_MPI_FUT_IRECV_SUBMIT_BEGIN, (src), (mpi_tag), _starpu_gettid());
#define _STARPU_MPI_TRACE_IRECV_SUBMIT_END(src, mpi_tag)	\
	FUT_DO_PROBE3(_STARPU_MPI_FUT_IRECV_SUBMIT_END, (src), (mpi_tag), _starpu_gettid());
#define _STARPU_MPI_TRACE_ISEND_COMPLETE_BEGIN(dest, mpi_tag, size)	\
	FUT_DO_PROBE4(_STARPU_MPI_FUT_ISEND_COMPLETE_BEGIN, (dest), (mpi_tag), (size), _starpu_gettid());
#define _STARPU_MPI_TRACE_ISEND_COMPLETE_END(dest, mpi_tag, size)	\
	FUT_DO_PROBE4(_STARPU_MPI_FUT_ISEND_COMPLETE_END, (dest), (mpi_tag), (size), _starpu_gettid());
#define _STARPU_MPI_TRACE_IRECV_COMPLETE_BEGIN(src, mpi_tag)	\
	FUT_DO_PROBE3(_STARPU_MPI_FUT_IRECV_COMPLETE_BEGIN, (src), (mpi_tag), _starpu_gettid());
#define _STARPU_MPI_TRACE_IRECV_COMPLETE_END(src, mpi_tag)	\
	FUT_DO_PROBE3(_STARPU_MPI_FUT_IRECV_COMPLETE_END, (src), (mpi_tag), _starpu_gettid());
#define _STARPU_MPI_TRACE_SLEEP_BEGIN()	\
	FUT_DO_PROBE1(_STARPU_MPI_FUT_SLEEP_BEGIN, _starpu_gettid());
#define _STARPU_MPI_TRACE_SLEEP_END()	\
	FUT_DO_PROBE1(_STARPU_MPI_FUT_SLEEP_END, _starpu_gettid());
#define _STARPU_MPI_TRACE_DTESTING_BEGIN()	\
	FUT_DO_PROBE1(_STARPU_MPI_FUT_DTESTING_BEGIN,  _starpu_gettid());
#define _STARPU_MPI_TRACE_DTESTING_END()	\
	FUT_DO_PROBE1(_STARPU_MPI_FUT_DTESTING_END, _starpu_gettid());
#define _STARPU_MPI_TRACE_UTESTING_BEGIN(src, mpi_tag)	\
	FUT_DO_PROBE3(_STARPU_MPI_FUT_UTESTING_BEGIN, (src), (mpi_tag),  _starpu_gettid());
#define _STARPU_MPI_TRACE_UTESTING_END(src, mpi_tag)	\
	FUT_DO_PROBE3(_STARPU_MPI_FUT_UTESTING_END, (src), (mpi_tag), _starpu_gettid());
#define _STARPU_MPI_TRACE_UWAIT_BEGIN(src, mpi_tag)	\
	FUT_DO_PROBE3(_STARPU_MPI_FUT_UWAIT_BEGIN, (src), (mpi_tag),  _starpu_gettid());
#define _STARPU_MPI_TRACE_UWAIT_END(src, mpi_tag)	\
	FUT_DO_PROBE3(_STARPU_MPI_FUT_UWAIT_END, (src), (mpi_tag), _starpu_gettid());
#define TRACE
#else
#define _STARPU_MPI_TRACE_START(a, b)				do {} while(0);
#define _STARPU_MPI_TRACE_STOP(a, b)				do {} while(0);
#define _STARPU_MPI_TRACE_BARRIER(a, b, c)			do {} while(0);
#define _STARPU_MPI_TRACE_ISEND_SUBMIT_BEGIN(a, b, c)		do {} while(0);
#define _STARPU_MPI_TRACE_ISEND_SUBMIT_END(a, b, c)		do {} while(0);
#define _STARPU_MPI_TRACE_IRECV_SUBMIT_BEGIN(a, b)		do {} while(0);
#define _STARPU_MPI_TRACE_IRECV_SUBMIT_END(a, b)		do {} while(0);
#define _STARPU_MPI_TRACE_ISEND_COMPLETE_BEGIN(a, b, c)		do {} while(0);
#define _STARPU_MPI_TRACE_ISEND_COMPLETE_END(a, b, c)		do {} while(0);
#define _STARPU_MPI_TRACE_IRECV_COMPLETE_BEGIN(a, b)		do {} while(0);
#define _STARPU_MPI_TRACE_IRECV_COMPLETE_END(a, b)		do {} while(0);
#define _STARPU_MPI_TRACE_SLEEP_BEGIN()				do {} while(0);
#define _STARPU_MPI_TRACE_SLEEP_END()				do {} while(0);
#define _STARPU_MPI_TRACE_DTESTING_BEGIN()			do {} while(0);
#define _STARPU_MPI_TRACE_DTESTING_END()			do {} while(0);
#define _STARPU_MPI_TRACE_UTESTING_BEGIN(a, b)			do {} while(0);
#define _STARPU_MPI_TRACE_UTESTING_END(a, b)			do {} while(0);
#define _STARPU_MPI_TRACE_UWAIT_BEGIN(a, b)			do {} while(0);
#define _STARPU_MPI_TRACE_UWAIT_END(a, b)			do {} while(0);
#endif

#ifdef __cplusplus
}
#endif


#endif // __STARPU_MPI_FXT_H__
