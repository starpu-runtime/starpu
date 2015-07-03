/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2012, 2014-2015  Université de Bordeaux
 * Copyright (C) 2010, 2011, 2012, 2013, 2015  CNRS
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

#include <starpu_mpi_private.h>

int _starpu_debug_rank=-1;
int _starpu_debug_level_min=0;
int _starpu_debug_level_max=0;
int _starpu_mpi_tag = 42;
int _starpu_mpi_comm;

void _starpu_mpi_set_debug_level_min(int level)
{
	_starpu_debug_level_min = level;
}

void _starpu_mpi_set_debug_level_max(int level)
{
	_starpu_debug_level_max = level;
}

int starpu_mpi_get_communication_tag(void)
{
	return _starpu_mpi_tag;
}

void starpu_mpi_set_communication_tag(int tag)
{
	_starpu_mpi_tag = tag;
}

char *_starpu_mpi_get_mpi_code(int code)
{
	switch (code)
	{
	case MPI_SUCCESS: return "MPI_SUCCESS";
#ifdef MPI_ERR_BUFFER
	case MPI_ERR_BUFFER: return "MPI_ERR_BUFFER";
#endif
#ifdef MPI_ERR_COUNT
	case MPI_ERR_COUNT: return "MPI_ERR_COUNT";
#endif
#ifdef MPI_ERR_TYPE
	case MPI_ERR_TYPE: return "MPI_ERR_TYPE";
#endif
#ifdef MPI_ERR_TAG
	case MPI_ERR_TAG: return "MPI_ERR_TAG";
#endif
#ifdef MPI_ERR_COMM
	case MPI_ERR_COMM: return "MPI_ERR_COMM";
#endif
#ifdef MPI_ERR_RANK
	case MPI_ERR_RANK: return "MPI_ERR_RANK";
#endif
#ifdef MPI_ERR_REQUEST
	case MPI_ERR_REQUEST: return "MPI_ERR_REQUEST";
#endif
#ifdef MPI_ERR_ROOT
	case MPI_ERR_ROOT: return "MPI_ERR_ROOT";
#endif
#ifdef MPI_ERR_GROUP
	case MPI_ERR_GROUP: return "MPI_ERR_GROUP";
#endif
#ifdef MPI_ERR_OP
	case MPI_ERR_OP: return "MPI_ERR_OP";
#endif
#ifdef MPI_ERR_TOPOLOGY
	case MPI_ERR_TOPOLOGY: return "MPI_ERR_TOPOLOGY";
#endif
#ifdef MPI_ERR_DIMS
	case MPI_ERR_DIMS: return "MPI_ERR_DIMS";
#endif
#ifdef MPI_ERR_ARG
	case MPI_ERR_ARG: return "MPI_ERR_ARG";
#endif
#ifdef MPI_ERR_UNKNOWN
	case MPI_ERR_UNKNOWN: return "MPI_ERR_UNKNOWN";
#endif
#ifdef MPI_ERR_TRUNCATE
	case MPI_ERR_TRUNCATE: return "MPI_ERR_TRUNCATE";
#endif
#ifdef MPI_ERR_OTHER
	case MPI_ERR_OTHER: return "MPI_ERR_OTHER";
#endif
#ifdef MPI_ERR_INTERN
	case MPI_ERR_INTERN: return "MPI_ERR_INTERN";
#endif
#ifdef MPI_ERR_IN_STATUS
	case MPI_ERR_IN_STATUS: return "MPI_ERR_IN_STATUS";
#endif
#ifdef MPI_ERR_PENDING
	case MPI_ERR_PENDING: return "MPI_ERR_PENDING";
#endif
#ifdef MPI_ERR_ACCESS
	case MPI_ERR_ACCESS: return "MPI_ERR_ACCESS";
#endif
#ifdef MPI_ERR_AMODE
	case MPI_ERR_AMODE: return "MPI_ERR_AMODE";
#endif
#ifdef MPI_ERR_ASSERT
	case MPI_ERR_ASSERT: return "MPI_ERR_ASSERT";
#endif
#ifdef MPI_ERR_BAD_FILE
	case MPI_ERR_BAD_FILE: return "MPI_ERR_BAD_FILE";
#endif
#ifdef MPI_ERR_BASE
	case MPI_ERR_BASE: return "MPI_ERR_BASE";
#endif
#ifdef MPI_ERR_CONVERSION
	case MPI_ERR_CONVERSION: return "MPI_ERR_CONVERSION";
#endif
#ifdef MPI_ERR_DISP
	case MPI_ERR_DISP: return "MPI_ERR_DISP";
#endif
#ifdef MPI_ERR_DUP_DATAREP
	case MPI_ERR_DUP_DATAREP: return "MPI_ERR_DUP_DATAREP";
#endif
#ifdef MPI_ERR_FILE_EXISTS
	case MPI_ERR_FILE_EXISTS: return "MPI_ERR_FILE_EXISTS";
#endif
#ifdef MPI_ERR_FILE_IN_USE
	case MPI_ERR_FILE_IN_USE: return "MPI_ERR_FILE_IN_USE";
#endif
#ifdef MPI_ERR_FILE
	case MPI_ERR_FILE: return "MPI_ERR_FILE";
#endif
#ifdef MPI_ERR_INFO_KEY
	case MPI_ERR_INFO_KEY: return "MPI_ERR_INFO_KEY";
#endif
#ifdef MPI_ERR_INFO_NOKEY
	case MPI_ERR_INFO_NOKEY: return "MPI_ERR_INFO_NOKEY";
#endif
#ifdef MPI_ERR_INFO_VALUE
	case MPI_ERR_INFO_VALUE: return "MPI_ERR_INFO_VALUE";
#endif
#ifdef MPI_ERR_INFO
	case MPI_ERR_INFO: return "MPI_ERR_INFO";
#endif
#ifdef MPI_ERR_IO
	case MPI_ERR_IO: return "MPI_ERR_IO";
#endif
#ifdef MPI_ERR_KEYVAL
	case MPI_ERR_KEYVAL: return "MPI_ERR_KEYVAL";
#endif
#ifdef MPI_ERR_LOCKTYPE
	case MPI_ERR_LOCKTYPE: return "MPI_ERR_LOCKTYPE";
#endif
#ifdef MPI_ERR_NAME
	case MPI_ERR_NAME: return "MPI_ERR_NAME";
#endif
#ifdef MPI_ERR_NO_MEM
	case MPI_ERR_NO_MEM: return "MPI_ERR_NO_MEM";
#endif
#ifdef MPI_ERR_NOT_SAME
	case MPI_ERR_NOT_SAME: return "MPI_ERR_NOT_SAME";
#endif
#ifdef MPI_ERR_NO_SPACE
	case MPI_ERR_NO_SPACE: return "MPI_ERR_NO_SPACE";
#endif
#ifdef MPI_ERR_NO_SUCH_FILE
	case MPI_ERR_NO_SUCH_FILE: return "MPI_ERR_NO_SUCH_FILE";
#endif
#ifdef MPI_ERR_PORT
	case MPI_ERR_PORT: return "MPI_ERR_PORT";
#endif
#ifdef MPI_ERR_QUOTA
	case MPI_ERR_QUOTA: return "MPI_ERR_QUOTA";
#endif
#ifdef MPI_ERR_READ_ONLY
	case MPI_ERR_READ_ONLY: return "MPI_ERR_READ_ONLY";
#endif
#ifdef MPI_ERR_RMA_CONFLICT
	case MPI_ERR_RMA_CONFLICT: return "MPI_ERR_RMA_CONFLICT";
#endif
#ifdef MPI_ERR_RMA_SYNC
	case MPI_ERR_RMA_SYNC: return "MPI_ERR_RMA_SYNC";
#endif
#ifdef MPI_ERR_SERVICE
	case MPI_ERR_SERVICE: return "MPI_ERR_SERVICE";
#endif
#ifdef MPI_ERR_SIZE
	case MPI_ERR_SIZE: return "MPI_ERR_SIZE";
#endif
#ifdef MPI_ERR_SPAWN
	case MPI_ERR_SPAWN: return "MPI_ERR_SPAWN";
#endif
#ifdef MPI_ERR_UNSUPPORTED_DATAREP
	case MPI_ERR_UNSUPPORTED_DATAREP: return "MPI_ERR_UNSUPPORTED_DATAREP";
#endif
#ifdef MPI_ERR_UNSUPPORTED_OPERATION
	case MPI_ERR_UNSUPPORTED_OPERATION: return "MPI_ERR_UNSUPPORTED_OPERATION";
#endif
#ifdef MPI_ERR_WIN
	case MPI_ERR_WIN: return "MPI_ERR_WIN";
#endif
#if defined(MPI_ERR_LASTCODE) && MPI_ERR_LASTCODE != MPI_SUCCESS
	case MPI_ERR_LASTCODE: return "MPI_ERR_LASTCODE";
#endif
	default: return "UNKNOWN_MPI_CODE";
	}
}
