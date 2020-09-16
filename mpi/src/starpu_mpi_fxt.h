/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2019       Federal University of Rio Grande do Sul (UFRGS)
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

/** @file */

#ifdef __cplusplus
extern "C"
{
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
#define _STARPU_MPI_FUT_DATA_SET_RANK			0x521a
#define _STARPU_MPI_FUT_IRECV_TERMINATED		0x521b
#define _STARPU_MPI_FUT_ISEND_TERMINATED		0x521c
#define _STARPU_MPI_FUT_TESTING_DETACHED_BEGIN		0x521d
#define _STARPU_MPI_FUT_TESTING_DETACHED_END		0x521e
#define _STARPU_MPI_FUT_TEST_BEGIN			0x521f
#define _STARPU_MPI_FUT_TEST_END			0x5220
#define _STARPU_MPI_FUT_IRECV_COMPLETE_BEGIN		0x520a
#define _STARPU_MPI_FUT_IRECV_COMPLETE_END		0x520b
#define _STARPU_MPI_FUT_SLEEP_BEGIN			0x520c
#define _STARPU_MPI_FUT_SLEEP_END			0x520d
#define _STARPU_MPI_FUT_DTESTING_BEGIN			0x520e
#define _STARPU_MPI_FUT_DTESTING_END			0x520f
#define _STARPU_MPI_FUT_UTESTING_BEGIN			0x5210
#define _STARPU_MPI_FUT_UTESTING_END			0x5211
#define _STARPU_MPI_FUT_UWAIT_BEGIN			0x5212
#define _STARPU_MPI_FUT_UWAIT_END			0x5213
#define _STARPU_MPI_FUT_POLLING_BEGIN			0x5214
#define _STARPU_MPI_FUT_POLLING_END			0x5215
#define _STARPU_MPI_FUT_DRIVER_RUN_BEGIN		0x5216
#define _STARPU_MPI_FUT_DRIVER_RUN_END			0x5217
#define _STARPU_MPI_FUT_DATA_SET_TAG			0x5218

#ifdef STARPU_USE_FXT

#define _STARPU_MPI_TRACE_START(rank, worldsize)	\
	FUT_DO_PROBE3(_STARPU_MPI_FUT_START, (rank), (worldsize), _starpu_gettid());
#define _STARPU_MPI_TRACE_STOP(rank, worldsize)	\
	FUT_DO_PROBE3(_STARPU_MPI_FUT_STOP, (rank), (worldsize), _starpu_gettid());
#define _STARPU_MPI_TRACE_BARRIER(rank, worldsize, key)	do {\
	if (_starpu_fxt_started) \
	FUT_DO_ALWAYS_PROBE4(_STARPU_MPI_FUT_BARRIER, (rank), (worldsize), (key), _starpu_gettid()); \
} while (0)
#define _STARPU_MPI_TRACE_ISEND_SUBMIT_BEGIN(dest, data_tag, size)	\
	FUT_DO_PROBE4(_STARPU_MPI_FUT_ISEND_SUBMIT_BEGIN, (dest), (data_tag), (size), _starpu_gettid());
#define _STARPU_MPI_TRACE_ISEND_SUBMIT_END(dest, data_tag, size, jobid)	\
	FUT_DO_PROBE5(_STARPU_MPI_FUT_ISEND_SUBMIT_END, (dest), (data_tag), (size), (jobid), _starpu_gettid());
#define _STARPU_MPI_TRACE_IRECV_SUBMIT_BEGIN(src, data_tag)	\
	FUT_DO_PROBE3(_STARPU_MPI_FUT_IRECV_SUBMIT_BEGIN, (src), (data_tag), _starpu_gettid());
#define _STARPU_MPI_TRACE_IRECV_SUBMIT_END(src, data_tag)	\
	FUT_DO_PROBE3(_STARPU_MPI_FUT_IRECV_SUBMIT_END, (src), (data_tag), _starpu_gettid());
#define _STARPU_MPI_TRACE_ISEND_COMPLETE_BEGIN(dest, data_tag, size)	\
	FUT_DO_PROBE4(_STARPU_MPI_FUT_ISEND_COMPLETE_BEGIN, (dest), (data_tag), (size), _starpu_gettid());
#define _STARPU_MPI_TRACE_COMPLETE_BEGIN(type, rank, data_tag)		\
	if (type == RECV_REQ) { _STARPU_MPI_TRACE_IRECV_COMPLETE_BEGIN((rank), (data_tag)); } else if (type == SEND_REQ) { _STARPU_MPI_TRACE_ISEND_COMPLETE_BEGIN((rank), (data_tag), 0); }
#define _STARPU_MPI_TRACE_ISEND_COMPLETE_END(dest, data_tag, size)	\
	FUT_DO_PROBE4(_STARPU_MPI_FUT_ISEND_COMPLETE_END, (dest), (data_tag), (size), _starpu_gettid());
#define _STARPU_MPI_TRACE_IRECV_COMPLETE_BEGIN(src, data_tag)	\
	FUT_DO_PROBE3(_STARPU_MPI_FUT_IRECV_COMPLETE_BEGIN, (src), (data_tag), _starpu_gettid());
#define _STARPU_MPI_TRACE_IRECV_COMPLETE_END(src, data_tag)	\
	FUT_DO_PROBE3(_STARPU_MPI_FUT_IRECV_COMPLETE_END, (src), (data_tag), _starpu_gettid());
#define _STARPU_MPI_TRACE_COMPLETE_END(type, rank, data_tag)		\
	if (type == RECV_REQ) { _STARPU_MPI_TRACE_IRECV_COMPLETE_END((rank), (data_tag)); } else if (type == SEND_REQ) { _STARPU_MPI_TRACE_ISEND_COMPLETE_END((rank), (data_tag), 0); }
#define _STARPU_MPI_TRACE_TERMINATED(req, rank, data_tag)		\
	if ((req)->request_type == RECV_REQ) FUT_DO_PROBE4(_STARPU_MPI_FUT_IRECV_TERMINATED, (rank), (data_tag), (req)->post_sync_jobid, _starpu_gettid()); else \
	if ((req)->request_type == SEND_REQ) FUT_DO_PROBE3(_STARPU_MPI_FUT_ISEND_TERMINATED, (rank), (data_tag), _starpu_gettid());
#define _STARPU_MPI_TRACE_SLEEP_BEGIN()	\
	FUT_DO_PROBE1(_STARPU_MPI_FUT_SLEEP_BEGIN, _starpu_gettid());
#define _STARPU_MPI_TRACE_SLEEP_END()	\
	FUT_DO_PROBE1(_STARPU_MPI_FUT_SLEEP_END, _starpu_gettid());
#define _STARPU_MPI_TRACE_DTESTING_BEGIN()	\
	FUT_DO_PROBE1(_STARPU_MPI_FUT_DTESTING_BEGIN,  _starpu_gettid());
#define _STARPU_MPI_TRACE_DTESTING_END()	\
	FUT_DO_PROBE1(_STARPU_MPI_FUT_DTESTING_END, _starpu_gettid());
#define _STARPU_MPI_TRACE_UTESTING_BEGIN(src, data_tag)	\
	FUT_DO_PROBE3(_STARPU_MPI_FUT_UTESTING_BEGIN, (src), (data_tag),  _starpu_gettid());
#define _STARPU_MPI_TRACE_UTESTING_END(src, data_tag)	\
	FUT_DO_PROBE3(_STARPU_MPI_FUT_UTESTING_END, (src), (data_tag), _starpu_gettid());
#define _STARPU_MPI_TRACE_UWAIT_BEGIN(src, data_tag)	\
	FUT_DO_PROBE3(_STARPU_MPI_FUT_UWAIT_BEGIN, (src), (data_tag),  _starpu_gettid());
#define _STARPU_MPI_TRACE_UWAIT_END(src, data_tag)	\
	FUT_DO_PROBE3(_STARPU_MPI_FUT_UWAIT_END, (src), (data_tag), _starpu_gettid());
#define _STARPU_MPI_TRACE_DATA_SET_RANK(handle, rank)	\
	FUT_DO_PROBE3(_STARPU_MPI_FUT_DATA_SET_RANK, (handle), (rank), _starpu_gettid());
#define _STARPU_MPI_TRACE_DATA_SET_TAG(handle, data_tag)	\
	FUT_DO_PROBE3(_STARPU_MPI_FUT_DATA_SET_TAG, (handle), (data_tag), _starpu_gettid());
#if 0
/* This is very expensive in the trace, only enable for debugging */
#define _STARPU_MPI_TRACE_TESTING_DETACHED_BEGIN()	\
	FUT_DO_PROBE1(_STARPU_MPI_FUT_TESTING_DETACHED_BEGIN, _starpu_gettid());
#define _STARPU_MPI_TRACE_TESTING_DETACHED_END()	\
	FUT_DO_PROBE1(_STARPU_MPI_FUT_TESTING_DETACHED_END, _starpu_gettid());
#define _STARPU_MPI_TRACE_TEST_BEGIN(peer, data_tag)	\
	FUT_DO_PROBE3(_STARPU_MPI_FUT_TEST_BEGIN, (peer), (data_tag), _starpu_gettid());
#define _STARPU_MPI_TRACE_TEST_END(peer, data_tag)	\
	FUT_DO_PROBE3(_STARPU_MPI_FUT_TEST_END, (peer), (data_tag), _starpu_gettid());
#else
#define _STARPU_MPI_TRACE_TESTING_DETACHED_BEGIN()		do {} while(0)
#define _STARPU_MPI_TRACE_TESTING_DETACHED_END()		do {} while(0)
#define _STARPU_MPI_TRACE_TEST_BEGIN(peer, data_tag)		do {} while(0)
#define _STARPU_MPI_TRACE_TEST_END(peer, data_tag)		do {} while(0)
#endif
#define _STARPU_MPI_TRACE_POLLING_BEGIN()					\
	if(!trace_loop) {						\
		trace_loop = 1;							\
		FUT_DO_PROBE1(_STARPU_MPI_FUT_POLLING_BEGIN, _starpu_gettid()); \
	}
#define _STARPU_MPI_TRACE_POLLING_END()	\
	if(trace_loop) {							\
		trace_loop = 0;							\
		FUT_DO_PROBE1(_STARPU_MPI_FUT_POLLING_END, _starpu_gettid());	\
	}
#define _STARPU_MPI_TRACE_DRIVER_RUN_BEGIN()	\
	FUT_DO_PROBE1(_STARPU_MPI_FUT_DRIVER_RUN_BEGIN,  _starpu_gettid());
#define _STARPU_MPI_TRACE_DRIVER_RUN_END()	\
	FUT_DO_PROBE1(_STARPU_MPI_FUT_DRIVER_RUN_END, _starpu_gettid());
#define _STARPU_MPI_TRACE_DRIVER_RUN_BEGIN()	\
	FUT_DO_PROBE1(_STARPU_MPI_FUT_DRIVER_RUN_BEGIN,  _starpu_gettid());
#define _STARPU_MPI_TRACE_DRIVER_RUN_END()	\
	FUT_DO_PROBE1(_STARPU_MPI_FUT_DRIVER_RUN_END, _starpu_gettid());
#define TRACE
#else
#define _STARPU_MPI_TRACE_START(a, b)				do {} while(0);
#define _STARPU_MPI_TRACE_STOP(a, b)				do {} while(0);
#define _STARPU_MPI_TRACE_BARRIER(a, b, c)			do {} while(0);
#define _STARPU_MPI_TRACE_ISEND_SUBMIT_BEGIN(a, b, c)		do {} while(0);
#define _STARPU_MPI_TRACE_ISEND_SUBMIT_END(a, b, c, d)		do {} while(0);
#define _STARPU_MPI_TRACE_IRECV_SUBMIT_BEGIN(a, b)		do {} while(0);
#define _STARPU_MPI_TRACE_IRECV_SUBMIT_END(a, b)		do {} while(0);
#define _STARPU_MPI_TRACE_ISEND_COMPLETE_BEGIN(a, b, c)		do {} while(0);
#define _STARPU_MPI_TRACE_COMPLETE_BEGIN(a, b, c)		do {} while(0);
#define _STARPU_MPI_TRACE_COMPLETE_END(a, b, c)			do {} while(0);
#define _STARPU_MPI_TRACE_TERMINATED(a, b, c)			do {} while(0);
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
#define _STARPU_MPI_TRACE_DATA_SET_RANK(a, b)			do {} while(0);
#define _STARPU_MPI_TRACE_DATA_SET_TAG(a, b)			do {} while(0);
#define _STARPU_MPI_TRACE_TESTING_DETACHED_BEGIN()		do {} while(0)
#define _STARPU_MPI_TRACE_TESTING_DETACHED_END()		do {} while(0)
#define _STARPU_MPI_TRACE_TEST_BEGIN(peer, data_tag)		do {} while(0)
#define _STARPU_MPI_TRACE_TEST_END(peer, data_tag)		do {} while(0)
#define _STARPU_MPI_TRACE_POLLING_BEGIN()			do {} while(0);
#define _STARPU_MPI_TRACE_POLLING_END()				do {} while(0);
#define _STARPU_MPI_TRACE_DRIVER_RUN_BEGIN()			do {} while(0);
#define _STARPU_MPI_TRACE_DRIVER_RUN_END()			do {} while(0);
#endif

#ifdef __cplusplus
}
#endif


#endif // __STARPU_MPI_FXT_H__
