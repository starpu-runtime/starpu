/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Corentin Salingue
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

#ifndef __STARPU_DISK_H__
#define __STARPU_DISK_H__

#include <sys/types.h>
#include <starpu_config.h>

/**
   @defgroup API_Out_Of_Core Out Of Core
   @{
*/

/**
   Set of functions to manipulate datas on disk.
*/
struct starpu_disk_ops
{
	/**
	   Connect a disk memory at location \p parameter with size \p size, and return a
	   base as void*, which will be passed by StarPU to all other methods.
	*/
	void *  (*plug)   (void *parameter, starpu_ssize_t size);
	/**
	   Disconnect a disk memory \p base.
	*/
	void    (*unplug) (void *base);

	/**
	   Measure the bandwidth and the latency for the disk \p node and save it. Returns
	   1 if it could measure it.
	*/
	int    (*bandwidth)    (unsigned node, void *base);

	/**
	   Create a new location for datas of size \p size. Return an opaque object pointer.
	*/
	void *  (*alloc)  (void *base, size_t size);

	/**
	   Free a data \p obj previously allocated with starpu_disk_ops::alloc.
	*/
	void    (*free)   (void *base, void *obj, size_t size);

	/**
	   Open an existing location of datas, at a specific position \p pos dependent on the backend.
	*/
	void *  (*open)   (void *base, void *pos, size_t size);
	/**
	   Close, without deleting it, a location of datas \p obj.
	*/
	void    (*close)  (void *base, void *obj, size_t size);

	/**
	   Read \p size bytes of data from \p obj in \p base, at offset \p offset, and put
	   into \p buf. Return the actual number of read bytes.
	*/
	int     (*read)   (void *base, void *obj, void *buf, off_t offset, size_t size);
	/**
	   Write \p size bytes of data to \p obj in \p base, at offset \p offset, from \p buf. Return 0 on success.
	*/
	int     (*write)  (void *base, void *obj, const void *buf, off_t offset, size_t size);

	/**
	   Read all data from \p obj of \p base, from offset 0. Returns it in an allocated buffer \p ptr, of size \p size
	*/
	int	(*full_read)    (void * base, void * obj, void ** ptr, size_t * size, unsigned dst_node);
	/**
	   Write data in \p ptr to \p obj of \p base, from offset 0, and truncate \p obj to
	   \p size, so that a \c full_read will get it.
	*/
	int 	(*full_write)   (void * base, void * obj, void * ptr, size_t size);

	/**
	   Asynchronously write \p size bytes of data to \p obj in \p base, at offset \p
	   offset, from \p buf. Return a void* pointer that StarPU will pass to \c
	   xxx_request methods for testing for the completion.
	*/
	void *  (*async_write)  (void *base, void *obj, void *buf, off_t offset, size_t size);
	/**
	   Asynchronously read \p size bytes of data from \p obj in \p base, at offset \p
	   offset, and put into \p buf. Return a void* pointer that StarPU will pass to \c
	   xxx_request methods for testing for the completion.
	*/
	void *  (*async_read)   (void *base, void *obj, void *buf, off_t offset, size_t size);

	/**
	   Read all data from \p obj of \p base, from offset 0. Return it in an allocated buffer \p ptr, of size \p size
	*/
	void *	(*async_full_read)    (void * base, void * obj, void ** ptr, size_t * size, unsigned dst_node);
	/**
	   Write data in \p ptr to \p obj of \p base, from offset 0, and truncate \p obj to
	   \p size, so that a starpu_disk_ops::full_read will get it.
	*/
	void *	(*async_full_write)   (void * base, void * obj, void * ptr, size_t size);

	/**
	   Copy from offset \p offset_src of disk object \p obj_src in \p base_src to
	   offset \p offset_dst of disk object \p obj_dst in \p base_dst. Return a void*
	   pointer that StarPU will pass to \c xxx_request methods for testing for the
	   completion.
	*/
	void *  (*copy)   (void *base_src, void* obj_src, off_t offset_src,  void *base_dst, void* obj_dst, off_t offset_dst, size_t size);

	/**
	   Wait for completion of request \p async_channel returned by a previous
	   asynchronous read, write or copy.
	*/
	void   (*wait_request) (void * async_channel);
	/**
	   Test for completion of request \p async_channel returned by a previous
	   asynchronous read, write or copy. Return 1 on completion, 0 otherwise.
	*/
	int    (*test_request) (void * async_channel);

	/**
	   Free the request allocated by a previous asynchronous read, write or copy.
	*/
	void   (*free_request)(void * async_channel);

	/* TODO: readv, writev, read2d, write2d, etc. */
};

/**
   Use the stdio library (fwrite, fread...) to read/write on disk.

   <strong>Warning: It creates one file per allocation !</strong>

   Do not support asynchronous transfers.
*/
extern struct starpu_disk_ops starpu_disk_stdio_ops;

/**
   Use the HDF5 library.

   <strong>It doesn't support multiple opening from different processes. </strong>

   You may only allow one process to write in the HDF5 file.

   <strong>If HDF5 library is not compiled with --thread-safe you can't open more than one HDF5 file at the same time. </strong>
*/
extern struct starpu_disk_ops starpu_disk_hdf5_ops;

/**
   Use the unistd library (write, read...) to read/write on disk.

   <strong>Warning: It creates one file per allocation !</strong>
*/
extern struct starpu_disk_ops starpu_disk_unistd_ops;

/**
   Use the unistd library (write, read...) to read/write on disk with the O_DIRECT flag.

   <strong>Warning: It creates one file per allocation !</strong>

   Only available on Linux systems.
*/
extern struct starpu_disk_ops starpu_disk_unistd_o_direct_ops;

/**
   Use the leveldb created by Google. More information at https://code.google.com/p/leveldb/
   Do not support asynchronous transfers.
*/
extern struct starpu_disk_ops starpu_disk_leveldb_ops;

/**
   Close an existing data opened with starpu_disk_open().
*/
void starpu_disk_close(unsigned node, void *obj, size_t size);

/**
   Open an existing file memory in a disk node. \p size is the size of
   the file. \p pos is the specific position dependent on the backend,
   given to the \c open  method of the disk operations. Return an
   opaque object pointer.
*/
void *starpu_disk_open(unsigned node, void *pos, size_t size);

/**
   Register a disk memory node with a set of functions to manipulate
   datas. The \c plug member of \p func will be passed \p parameter,
   and return a \c base which will be passed to all \p func methods.
   <br />
   SUCCESS: return the disk node. <br />
   FAIL: return an error code. <br />
   \p size must be at least \ref STARPU_DISK_SIZE_MIN bytes ! \p size
   being negative means infinite size.
*/
int starpu_disk_register(struct starpu_disk_ops *func, void *parameter, starpu_ssize_t size);

/**
   Minimum size of a registered disk. The size of a disk is the last
   parameter of the function starpu_disk_register().
*/
#define STARPU_DISK_SIZE_MIN (16*1024*1024)

/**
   Contain the node number of the disk swap, if set up through the
   \ref STARPU_DISK_SWAP variable.
*/
extern int starpu_disk_swap_node;

/** @} */

#endif /* __STARPU_DISK_H__ */
