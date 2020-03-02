/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2010       Mehdi Juhoor
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

#ifndef __STARPU_DATA_FILTERS_H__
#define __STARPU_DATA_FILTERS_H__

#include <starpu.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
   @defgroup API_Data_Partition Data Partition
   @{
*/

struct starpu_data_interface_ops;

/**
   Describe a data partitioning operation, to be given to starpu_data_partition()
*/
struct starpu_data_filter
{
	/**
	   Fill the \p child_interface structure with interface information
	   for the \p i -th child of the parent \p father_interface (among
	   \p nparts). The \p filter structure is provided, allowing to inspect the
	   starpu_data_filter::filter_arg and starpu_data_filter::filter_arg_ptr
	   parameters.
	   The details of what needs to be filled in \p child_interface vary according
	   to the data interface, but generally speaking:
	   <ul>
	   <li> <c>id</c> is usually just copied over from the father,
	   when the sub data has the same structure as the father,
	   e.g. a subvector is a vector, a submatrix is a matrix, etc.
	   This is however not the case for instance when dividing a
	   BCSR matrix into its dense blocks, which then are matrices.
	   </li>
	   <li> <c>nx</c>, <c>ny</c> and alike are usually divided by
	   the number of subdata, depending how the subdivision is
	   done (e.g. nx division vs ny division for vertical matrix
	   division vs horizontal matrix division). </li>
	   <li> <c>ld</c> for matrix interfaces are usually just
	   copied over: the leading dimension (ld) usually does not
	   change. </li>
	   <li> <c>elemsize</c> is usually just copied over. </li>
	   <li> <c>ptr</c>, the pointer to the data, has to be
	   computed according to \p i and the father's <c>ptr</c>, so
	   as to point to the start of the sub data. This should
	   however be done only if the father has <c>ptr</c> different
	   from NULL: in the OpenCL case notably, the
	   <c>dev_handle</c> and <c>offset</c> fields are used
	   instead. </li>
	   <li> <c>dev_handle</c> should be just copied over from the
	   parent. </li>
	   <li> <c>offset</c> has to be computed according to \p i and
	   the father's <c>offset</c>, so as to provide the offset of
	   the start of the sub data. This is notably used for the
	   OpenCL case.
	   </ul>
	*/
	void (*filter_func)(void *father_interface, void *child_interface, struct starpu_data_filter *, unsigned id, unsigned nparts);
	unsigned nchildren; /**< Number of parts to partition the data into. */
	/**
	   Return the number of children. This can be used instead of
	   starpu_data_filter::nchildren when the number of children depends
	   on the actual data (e.g. the number of blocks in a sparse
	   matrix).
	*/
	unsigned (*get_nchildren)(struct starpu_data_filter *, starpu_data_handle_t initial_handle);
	/**
	   When children use different data interface,
	   return which interface is used by child number \p id.
	*/
	struct starpu_data_interface_ops *(*get_child_ops)(struct starpu_data_filter *, unsigned id);
	unsigned filter_arg; /**< Additional parameter for the filter function */
	/**
	   Additional pointer parameter for
	   the filter function, such as the
	   sizes of the different parts. */
	void *filter_arg_ptr;
};

/**
   @name Basic API
   @{
*/

/**
   Request the partitioning of \p initial_handle into several subdata
   according to the filter \p f.

   Here an example of how to use the function.
   \code{.c}
   struct starpu_data_filter f =
   {
     .filter_func = starpu_matrix_filter_block,
     .nchildren = nslicesx
   };
   starpu_data_partition(A_handle, &f);
    \endcode
*/
void starpu_data_partition(starpu_data_handle_t initial_handle, struct starpu_data_filter *f);

/**
  Unapply the filter which has been applied to \p root_data, thus
  unpartitioning the data. The pieces of data are collected back into
  one big piece in the \p gathering_node (usually ::STARPU_MAIN_RAM).
  Tasks working on the partitioned data will be waited for
  by starpu_data_unpartition().

  Here an example of how to use the function.
  \code{.c}
  starpu_data_unpartition(A_handle, STARPU_MAIN_RAM);
  \endcode
*/
void starpu_data_unpartition(starpu_data_handle_t root_data, unsigned gathering_node);

/**
   Return the \p i -th child of the given \p handle, which must have
   been partitionned beforehand.
*/
starpu_data_handle_t starpu_data_get_child(starpu_data_handle_t handle, unsigned i);

/**
   Return the number of children \p handle has been partitioned into.
*/
int starpu_data_get_nb_children(starpu_data_handle_t handle);

/**
   After partitioning a StarPU data by applying a filter,
   starpu_data_get_sub_data() can be used to get handles for each of the
   data portions. \p root_data is the parent data that was partitioned.
   \p depth is the number of filters to traverse (in case several filters
   have been applied, to e.g. partition in row blocks, and then in column
   blocks), and the subsequent parameters are the indexes. The function
   returns a handle to the subdata.

   Here an example of how to use the function.
   \code{.c}
   h = starpu_data_get_sub_data(A_handle, 1, taskx);
   \endcode
*/
starpu_data_handle_t starpu_data_get_sub_data(starpu_data_handle_t root_data, unsigned depth, ... );

/**
   Similar to starpu_data_get_sub_data() but use a \c va_list for the
   parameter list.
*/
starpu_data_handle_t starpu_data_vget_sub_data(starpu_data_handle_t root_data, unsigned depth, va_list pa);

/**
   Apply \p nfilters filters to the handle designated by \p
   root_handle recursively. \p nfilters pointers to variables of the
   type starpu_data_filter should be given.
*/
void starpu_data_map_filters(starpu_data_handle_t root_data, unsigned nfilters, ...);

/**
   Apply \p nfilters filters to the handle designated by
   \p root_handle recursively. Use a \p va_list of pointers to
   variables of the type starpu_data_filter.
*/
void starpu_data_vmap_filters(starpu_data_handle_t root_data, unsigned nfilters, va_list pa);

/** @} */

/**
   @name Asynchronous API
   @{
*/

/**
   Plan to partition \p initial_handle into several subdata according to
   the filter \p f.
   The handles are returned into the \p children array, which has to be
   the same size as the number of parts described in \p f. These handles
   are not immediately usable, starpu_data_partition_submit() has to be
   called to submit the actual partitioning.

   Here is an example of how to use the function:
   \code{.c}
   starpu_data_handle_t children[nslicesx];
   struct starpu_data_filter f =
   {
     .filter_func = starpu_matrix_filter_block,
     .nchildren = nslicesx
     };
     starpu_data_partition_plan(A_handle, &f, children);
\endcode
*/
void starpu_data_partition_plan(starpu_data_handle_t initial_handle, struct starpu_data_filter *f, starpu_data_handle_t *children);

/**
   Submit the actual partitioning of \p initial_handle into the \p nparts
   \p children handles. This call is asynchronous, it only submits that the
   partitioning should be done, so that the \p children handles can now be used to
   submit tasks, and \p initial_handle can not be used to submit tasks any more (to
   guarantee coherency).
   For instance,
   \code{.c}
   starpu_data_partition_submit(A_handle, nslicesx, children);
   \endcode
*/
void starpu_data_partition_submit(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children);

/**
   Similar to starpu_data_partition_submit(), but do not invalidate \p
   initial_handle. This allows to continue using it, but the application has to be
   careful not to write to \p initial_handle or \p children handles, only read from
   them, since the coherency is otherwise not guaranteed.  This thus allows to
   submit various tasks which concurrently read from various partitions of the data.

   When the application wants to write to \p initial_handle again, it should call
   starpu_data_unpartition_submit(), which will properly add dependencies between the
   reads on the \p children and the writes to be submitted.

   If instead the application wants to write to \p children handles, it should
   call starpu_data_partition_readwrite_upgrade_submit(), which will correctly add
   dependencies between the reads on the \p initial_handle and the writes to be
   submitted.
*/
void starpu_data_partition_readonly_submit(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children);

/**
   Assume that a partitioning of \p initial_handle has already been submited
   in readonly mode through starpu_data_partition_readonly_submit(), and will upgrade
   that partitioning into read-write mode for the \p children, by invalidating \p
   initial_handle, and adding the necessary dependencies.
*/
void starpu_data_partition_readwrite_upgrade_submit(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children);

/**
   Assuming that \p initial_handle is partitioned into \p children,
   submit an unpartitionning of \p initial_handle, i.e. submit a
   gathering of the pieces on the requested \p gathering_node memory
   node, and submit an invalidation of the children.
 */
void starpu_data_unpartition_submit(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children, int gathering_node);

void starpu_data_unpartition_submit_r(starpu_data_handle_t initial_handle, int gathering_node);

/**
   Similar to starpu_data_partition_submit(), but do not invalidate \p
   initial_handle. This allows to continue using it, but the application has to be
   careful not to write to \p initial_handle or \p children handles, only read from
   them, since the coherency is otherwise not guaranteed.  This thus allows to
   submit various tasks which concurrently read from various
   partitions of the data.
*/
void starpu_data_unpartition_readonly_submit(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children, int gathering_node);

/**
   Clear the partition planning established between \p root_data and
   \p children with starpu_data_partition_plan(). This will notably
   submit an unregister all the \p children, which can thus not be
   used any more afterwards.
*/
void starpu_data_partition_clean(starpu_data_handle_t root_data, unsigned nparts, starpu_data_handle_t *children);

/**
   Similar to starpu_data_unpartition_submit_sequential_consistency()
   but allow to specify a callback function for the unpartitiong task
*/
void starpu_data_unpartition_submit_sequential_consistency_cb(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children, int gather_node, int sequential_consistency, void (*callback_func)(void *), void *callback_arg);

/**
   Similar to starpu_data_partition_submit() but also allow to specify
   the coherency to be used for the main data \p initial_handle
   through the parameter \p sequential_consistency.
*/
void starpu_data_partition_submit_sequential_consistency(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children, int sequential_consistency);

/**
   Similar to starpu_data_unpartition_submit() but also allow to specify
   the coherency to be used for the main data \p initial_handle
   through the parameter \p sequential_consistency.
*/
void starpu_data_unpartition_submit_sequential_consistency(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children, int gathering_node, int sequential_consistency);

/**
   Disable the automatic partitioning of the data \p handle for which
   a asynchronous plan has previously been submitted
*/
void starpu_data_partition_not_automatic(starpu_data_handle_t handle);

/** @} */

/**
   @name Predefined BCSR Filter Functions
   Predefined partitioning functions for BCSR data. Examples on how to
   use them are shown in \ref PartitioningData.
   @{
*/

/**
   Partition a block-sparse matrix into dense matrices.
   starpu_data_filter::get_child_ops needs to be set to
   starpu_bcsr_filter_canonical_block_child_ops()
*/
void starpu_bcsr_filter_canonical_block(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);
/**
   Return the child_ops of the partition obtained with starpu_bcsr_filter_canonical_block().
*/
struct starpu_data_interface_ops *starpu_bcsr_filter_canonical_block_child_ops(struct starpu_data_filter *f, unsigned child);

/**
   Partition a block-sparse matrix into block-sparse matrices.

   The split is done along the leading dimension, i.e. along adjacent nnz blocks.
*/
void starpu_bcsr_filter_vertical_block(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/** @} */

/**
   @name Predefined CSR Filter Functions
   Predefined partitioning functions for CSR data. Examples on how to
   use them are shown in \ref PartitioningData.
   @{
*/

/**
   Partition a block-sparse matrix into vertical block-sparse matrices.
*/
void starpu_csr_filter_vertical_block(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/** @} */

/**
   @name Predefined Matrix Filter Functions
   Predefined partitioning functions for matrix
   data. Examples on how to use them are shown in \ref
   PartitioningData.
   Note: this is using the C element order which is row-major, i.e. elements
   with consecutive x coordinates are consecutive in memory.
   @{
*/

/**
   Partition a dense Matrix along the x dimension, thus getting (x/\p
   nparts ,y) matrices. If \p nparts does not divide x, the last
   submatrix contains the remainder.
*/
void starpu_matrix_filter_block(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Partition a dense Matrix along the x dimension, with a
   shadow border <c>filter_arg_ptr</c>, thus getting ((x-2*shadow)/\p
   nparts +2*shadow,y) matrices. If \p nparts does not divide x-2*shadow,
   the last submatrix contains the remainder.

   <b>IMPORTANT</b>: This can
   only be used for read-only access, as no coherency is enforced for the
   shadowed parts. A usage example is available in
   examples/filters/shadow2d.c
*/
void starpu_matrix_filter_block_shadow(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Partition a dense Matrix along the y dimension, thus getting
   (x,y/\p nparts) matrices. If \p nparts does not divide y, the last
   submatrix contains the remainder.
*/
void starpu_matrix_filter_vertical_block(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Partition a dense Matrix along the y dimension, with a
   shadow border <c>filter_arg_ptr</c>, thus getting
   (x,(y-2*shadow)/\p nparts +2*shadow) matrices. If \p nparts does not
   divide y-2*shadow, the last submatrix contains the remainder.

   <b>IMPORTANT</b>: This can only be used for read-only access, as no
   coherency is enforced for the shadowed parts. A usage example is
   available in examples/filters/shadow2d.c
*/
void starpu_matrix_filter_vertical_block_shadow(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/** @} */

/**
   @name Predefined Vector Filter Functions
   Predefined partitioning functions for vector
   data. Examples on how to use them are shown in \ref
   PartitioningData.
   @{
*/

/**
   Return in \p child_interface the \p id th element of the vector
   represented by \p father_interface once partitioned in \p nparts chunks of
   equal size.
*/
void starpu_vector_filter_block(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Return in \p child_interface the \p id th element of the vector
   represented by \p father_interface once partitioned in \p nparts chunks of
   equal size with a shadow border <c>filter_arg_ptr</c>, thus getting a vector
   of size <c>(n-2*shadow)/nparts+2*shadow</c>. The <c>filter_arg_ptr</c> field
   of \p f must be the shadow size casted into \c void*.

   <b>IMPORTANT</b>: This can only be used for read-only access, as no coherency is
   enforced for the shadowed parts. An usage example is available in
   examples/filters/shadow.c
*/
void starpu_vector_filter_block_shadow(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Return in \p child_interface the \p id th element of the vector
   represented by \p father_interface once partitioned into \p nparts chunks
   according to the <c>filter_arg_ptr</c> field of \p f. The
   <c>filter_arg_ptr</c> field must point to an array of \p nparts long
   elements, each of which specifies the number of elements in each chunk
   of the partition.
*/
void starpu_vector_filter_list_long(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
  Return in \p child_interface the \p id th element of the vector
  represented by \p father_interface once partitioned into \p nparts chunks
  according to the <c>filter_arg_ptr</c> field of \p f. The
  <c>filter_arg_ptr</c> field must point to an array of \p nparts uint32_t
  elements, each of which specifies the number of elements in each chunk
  of the partition.
*/
void starpu_vector_filter_list(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Return in \p child_interface the \p id th element of the vector
   represented by \p father_interface once partitioned in <c>2</c> chunks of
   equal size, ignoring nparts. Thus, \p id must be <c>0</c> or <c>1</c>.
*/
void starpu_vector_filter_divide_in_2(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/** @} */

/**
   @name Predefined Block Filter Functions
   Predefined partitioning functions for block data. Examples on how
   to use them are shown in \ref PartitioningData. An example is
   available in \c examples/filters/shadow3d.c
   Note: this is using the C element order which is row-major, i.e. elements
   with consecutive x coordinates are consecutive in memory.
   @{
*/

/**
  Partition a block along the X dimension, thus getting
  (x/\p nparts ,y,z) 3D matrices. If \p nparts does not divide x, the last
  submatrix contains the remainder.
 */
void starpu_block_filter_block(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Partition a block along the X dimension, with a
   shadow border <c>filter_arg_ptr</c>, thus getting
   ((x-2*shadow)/\p nparts +2*shadow,y,z) blocks. If \p nparts does not
   divide x, the last submatrix contains the remainder.

   <b>IMPORTANT</b>:
   This can only be used for read-only access, as no coherency is
   enforced for the shadowed parts.
*/
void starpu_block_filter_block_shadow(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Partition a block along the Y dimension, thus getting
   (x,y/\p nparts ,z) blocks. If \p nparts does not divide y, the last
   submatrix contains the remainder.
 */
void starpu_block_filter_vertical_block(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Partition a block along the Y dimension, with a
   shadow border <c>filter_arg_ptr</c>, thus getting
   (x,(y-2*shadow)/\p nparts +2*shadow,z) 3D matrices. If \p nparts does not
   divide y, the last submatrix contains the remainder.

   <b>IMPORTANT</b>:
   This can only be used for read-only access, as no coherency is
   enforced for the shadowed parts.
*/
void starpu_block_filter_vertical_block_shadow(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Partition a block along the Z dimension, thus getting
   (x,y,z/\p nparts) blocks. If \p nparts does not divide z, the last
   submatrix contains the remainder.
*/
void starpu_block_filter_depth_block(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Partition a block along the Z dimension, with a
   shadow border <c>filter_arg_ptr</c>, thus getting
   (x,y,(z-2*shadow)/\p nparts +2*shadow) blocks. If \p nparts does not
   divide z, the last submatrix contains the remainder.

   <b>IMPORTANT</b>:
   This can only be used for read-only access, as no coherency is
   enforced for the shadowed parts.
*/
void starpu_block_filter_depth_block_shadow(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Given an integer \p n, \p n the number of parts it must be divided in, \p id the
   part currently considered, determines the \p chunk_size and the \p offset, taking
   into account the size of the elements stored in the data structure \p elemsize
   and \p ld, the leading dimension, which is most often 1.
 */
void
starpu_filter_nparts_compute_chunk_size_and_offset(unsigned n, unsigned nparts,
					     size_t elemsize, unsigned id,
					     unsigned ld, unsigned *chunk_size,
					     size_t *offset);

/** @} */

/** @} */

#ifdef __cplusplus
}
#endif

#endif
