/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
extern "C" {
#endif

/**
   @defgroup API_Data_Partition Data Partition
   @{
*/

struct starpu_data_interface_ops;

/**
   Describe a data partitioning operation, to be given to starpu_data_partition(). 
   See \ref DefiningANewDataFilter for more details.
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

   See \ref PartitioningData for more details.
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

  See \ref PartitioningData for more details.
*/
void starpu_data_unpartition(starpu_data_handle_t root_data, unsigned gathering_node);

/**
   Return the \p i -th child of the given \p handle, which must have
   been partitionned beforehand.
   See \ref PartitioningData for more details.
*/
starpu_data_handle_t starpu_data_get_child(starpu_data_handle_t handle, unsigned i);

/**
   Return the number of children \p handle has been partitioned into.
   See \ref PartitioningData for more details.
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

   See \ref PartitioningData for more details.
*/
starpu_data_handle_t starpu_data_get_sub_data(starpu_data_handle_t root_data, unsigned depth, ...);

/**
   Similar to starpu_data_get_sub_data() but use a \c va_list for the
   parameter list.
   See \ref PartitioningData for more details.
*/
starpu_data_handle_t starpu_data_vget_sub_data(starpu_data_handle_t root_data, unsigned depth, va_list pa);

/**
   Apply \p nfilters filters to the handle designated by \p
   root_handle recursively. \p nfilters pointers to variables of the
   type starpu_data_filter should be given.
   See \ref PartitioningData for more details.
*/
void starpu_data_map_filters(starpu_data_handle_t root_data, unsigned nfilters, ...);

/**
   Apply \p nfilters filters to the handle designated by
   \p root_handle recursively. Use a \p va_list of pointers to
   variables of the type starpu_data_filter.
   See \ref PartitioningData for more details.
*/
void starpu_data_vmap_filters(starpu_data_handle_t root_data, unsigned nfilters, va_list pa);

/**
   Apply \p nfilters filters to the handle designated by \p
   root_handle recursively. The pointer of the filter list \p filters
   of the type starpu_data_filter should be given.
   See \ref PartitioningData for more details.
*/
void starpu_data_map_filters_parray(starpu_data_handle_t root_handle, int nfilters, struct starpu_data_filter **filters);

/**
   Apply \p nfilters filters to the handle designated by \p
   root_handle recursively. The list of filter \p filters
   of the type starpu_data_filter should be given.
   See \ref PartitioningData for more details.
*/
void starpu_data_map_filters_array(starpu_data_handle_t root_handle, int nfilters, struct starpu_data_filter *filters);

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

   See \ref AsynchronousPartitioning for more details.
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

   See \ref AsynchronousPartitioning for more details.
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
   See \ref AsynchronousPartitioning for more details.
*/
void starpu_data_partition_readonly_submit(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children);

/**
   Similar to starpu_data_partition_readonly_submit(), but allow to
   specify the coherency to be used for the main data \p initial_handle.
   See \ref AsynchronousPartitioning for more details.
 */
void starpu_data_partition_readonly_submit_sequential_consistency(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children, int sequential_consistency);

/**
   Assume that a partitioning of \p initial_handle has already been submited
   in readonly mode through starpu_data_partition_readonly_submit(), and will upgrade
   that partitioning into read-write mode for the \p children, by invalidating \p
   initial_handle, and adding the necessary dependencies.
   See \ref AsynchronousPartitioning for more details.
*/
void starpu_data_partition_readwrite_upgrade_submit(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children);

/**
   Assuming that \p initial_handle is partitioned into \p children,
   submit an unpartitionning of \p initial_handle, i.e. submit a
   gathering of the pieces on the requested \p gathering_node memory
   node, and submit an invalidation of the children.
   See \ref AsynchronousPartitioning for more details.
 */
void starpu_data_unpartition_submit(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children, int gathering_node);

/**
   Similar to starpu_data_partition_submit(), but do not invalidate \p
   initial_handle. This allows to continue using it, but the application has to be
   careful not to write to \p initial_handle or \p children handles, only read from
   them, since the coherency is otherwise not guaranteed.  This thus allows to
   submit various tasks which concurrently read from various
   partitions of the data.
   See \ref AsynchronousPartitioning for more details.
*/
void starpu_data_unpartition_readonly_submit(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children, int gathering_node);

/**
   Clear the partition planning established between \p root_data and
   \p children with starpu_data_partition_plan(). This will notably
   submit an unregister all the \p children, which can thus not be
   used any more afterwards.
   See \ref AsynchronousPartitioning for more details.
*/
void starpu_data_partition_clean(starpu_data_handle_t root_data, unsigned nparts, starpu_data_handle_t *children);

/**
   Similar to starpu_data_partition_clean() but the root data will be
   gathered on the given node.
   See \ref AsynchronousPartitioning for more details.
*/
void starpu_data_partition_clean_node(starpu_data_handle_t root_data, unsigned nparts, starpu_data_handle_t *children, int gather_node);

/**
   Similar to starpu_data_unpartition_submit_sequential_consistency()
   but allow to specify a callback function for the unpartitiong task.
   See \ref AsynchronousPartitioning for more details.
*/
void starpu_data_unpartition_submit_sequential_consistency_cb(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children, int gather_node, int sequential_consistency, void (*callback_func)(void *), void *callback_arg);

/**
   Similar to starpu_data_partition_submit() but also allow to specify
   the coherency to be used for the main data \p initial_handle
   through the parameter \p sequential_consistency.
   See \ref AsynchronousPartitioning for more details.
*/
void starpu_data_partition_submit_sequential_consistency(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children, int sequential_consistency);

/**
   Similar to starpu_data_unpartition_submit() but also allow to specify
   the coherency to be used for the main data \p initial_handle
   through the parameter \p sequential_consistency.
   See \ref AsynchronousPartitioning for more details.
*/
void starpu_data_unpartition_submit_sequential_consistency(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children, int gathering_node, int sequential_consistency);

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
   and starpu_data_filter::get_nchildren set to
   starpu_bcsr_filter_canonical_block_get_nchildren().

   See \ref BCSRDataInterface for more details.
*/
void starpu_bcsr_filter_canonical_block(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Return the number of children obtained with starpu_bcsr_filter_canonical_block().
   See \ref BCSRDataInterface for more details.
*/
unsigned starpu_bcsr_filter_canonical_block_get_nchildren(struct starpu_data_filter *f, starpu_data_handle_t handle);
/**
   Return the child_ops of the partition obtained with starpu_bcsr_filter_canonical_block().
   See \ref BCSRDataInterface for more details.
*/
struct starpu_data_interface_ops *starpu_bcsr_filter_canonical_block_child_ops(struct starpu_data_filter *f, unsigned child);

/**
   Partition a block-sparse matrix into block-sparse matrices.

   The split is done along the leading dimension, i.e. along adjacent nnz blocks.

   See \ref BCSRDataInterface for more details.
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

   See \ref CSRDataInterface for more details.
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

   See \ref MatrixDataInterface for more details.
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

   See \ref MatrixDataInterface for more details.
*/
void starpu_matrix_filter_block_shadow(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Partition a dense Matrix along the y dimension, thus getting
   (x,y/\p nparts) matrices. If \p nparts does not divide y, the last
   submatrix contains the remainder.

   See \ref MatrixDataInterface for more details.
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

   See \ref MatrixDataInterface for more details.
*/
void starpu_matrix_filter_vertical_block_shadow(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Pick \p nparts contiguous vectors from a matrix along
   the Y dimension. The starting position on Y-axis is set in
   <c>starpu_data_filter::filter_arg_ptr</c>.

   <c>starpu_data_filter::get_child_ops</c> needs to be set to
   starpu_matrix_filter_pick_vector_child_ops. A usage example is
   available in examples/filters/fmatrix_pick_vector.c

   See \ref MatrixDataInterface for more details.
*/
void starpu_matrix_filter_pick_vector_y(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Return the child_ops of the partition obtained with starpu_matrix_filter_pick_vector_y().
   See \ref MatrixDataInterface for more details.
*/
struct starpu_data_interface_ops *starpu_matrix_filter_pick_vector_child_ops(struct starpu_data_filter *f, unsigned child);

/**
   Pick \p nparts contiguous variables from a matrix. The starting position
   is set in <c>starpu_data_filter::filter_arg_ptr</c>.

   <c>starpu_data_filter::get_child_ops</c> needs to be set to
   starpu_matrix_filter_pick_variable_child_ops. A usage example is
   available in examples/filters/fmatrix_pick_variable.c

   See \ref MatrixDataInterface for more details.
*/
void starpu_matrix_filter_pick_variable(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Return the child_ops of the partition obtained with starpu_matrix_filter_pick_variable().
   See \ref MatrixDataInterface for more details.
*/
struct starpu_data_interface_ops *starpu_matrix_filter_pick_variable_child_ops(struct starpu_data_filter *f, unsigned child);

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

   See \ref VectorDataInterface for more details.
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

   See \ref VectorDataInterface for more details.
*/
void starpu_vector_filter_block_shadow(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Return in \p child_interface the \p id th element of the vector
   represented by \p father_interface once partitioned into \p nparts chunks
   according to the <c>filter_arg_ptr</c> field of \p f. The
   <c>filter_arg_ptr</c> field must point to an array of \p nparts long
   elements, each of which specifies the number of elements in each chunk
   of the partition.

   See \ref VectorDataInterface for more details.
*/
void starpu_vector_filter_list_long(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
  Return in \p child_interface the \p id th element of the vector
  represented by \p father_interface once partitioned into \p nparts chunks
  according to the <c>filter_arg_ptr</c> field of \p f. The
  <c>filter_arg_ptr</c> field must point to an array of \p nparts uint32_t
  elements, each of which specifies the number of elements in each chunk
  of the partition.

  See \ref VectorDataInterface for more details.
*/
void starpu_vector_filter_list(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Return in \p child_interface the \p id th element of the vector
   represented by \p father_interface once partitioned in <c>2</c> chunks of
   equal size, ignoring nparts. Thus, \p id must be <c>0</c> or <c>1</c>.

   See \ref VectorDataInterface for more details.
*/
void starpu_vector_filter_divide_in_2(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Pick \p nparts contiguous variables from a vector. The starting
   position is set in <c>starpu_data_filter::filter_arg_ptr</c>.

   <c>starpu_data_filter::get_child_ops</c> needs to be set to
   starpu_vector_filter_pick_variable_child_ops. A usage example is
   available in examples/filters/fvector_pick_variable.c

   See \ref VectorDataInterface for more details.
*/
void starpu_vector_filter_pick_variable(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Return the child_ops of the partition obtained with starpu_vector_filter_pick_variable().
   See \ref VectorDataInterface for more details.
*/
struct starpu_data_interface_ops *starpu_vector_filter_pick_variable_child_ops(struct starpu_data_filter *f, unsigned child);

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

  See \ref BlockDataInterface for more details.
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

   See \ref BlockDataInterface for more details.
*/
void starpu_block_filter_block_shadow(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Partition a block along the Y dimension, thus getting
   (x,y/\p nparts ,z) blocks. If \p nparts does not divide y, the last
   submatrix contains the remainder.

   See \ref BlockDataInterface for more details.
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

   See \ref BlockDataInterface for more details.
*/
void starpu_block_filter_vertical_block_shadow(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Partition a block along the Z dimension, thus getting
   (x,y,z/\p nparts) blocks. If \p nparts does not divide z, the last
   submatrix contains the remainder.

   See \ref BlockDataInterface for more details.
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

   See \ref BlockDataInterface for more details.
*/
void starpu_block_filter_depth_block_shadow(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Pick \p nparts contiguous matrices from a block along
   the Z dimension. The starting position on Z-axis is set in
   <c>starpu_data_filter::filter_arg_ptr</c>.

   <c>starpu_data_filter::get_child_ops</c> needs to be set to
   starpu_block_filter_pick_matrix_child_ops. A usage example is
   available in examples/filters/fblock_pick_matrix.c

   See \ref BlockDataInterface for more details.
*/
void starpu_block_filter_pick_matrix_z(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Pick \p nparts contiguous matrices from a block along
   the Y dimension. The starting position on Y-axis is set in
   <c>starpu_data_filter::filter_arg_ptr</c>.

   <c>starpu_data_filter::get_child_ops</c> needs to be set to
   starpu_block_filter_pick_matrix_child_ops. A usage example is
   available in examples/filters/fblock_pick_matrix.c

   See \ref BlockDataInterface for more details.
*/
void starpu_block_filter_pick_matrix_y(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Return the child_ops of the partition obtained with starpu_block_filter_pick_matrix_z()
   and starpu_block_filter_pick_matrix_y().
   See \ref BlockDataInterface for more details.
*/
struct starpu_data_interface_ops *starpu_block_filter_pick_matrix_child_ops(struct starpu_data_filter *f, unsigned child);

/**
   Pick \p nparts contiguous variables from a block. The starting position
   is set in <c>starpu_data_filter::filter_arg_ptr</c>.

   <c>starpu_data_filter::get_child_ops</c> needs to be set to
   starpu_block_filter_pick_variable_child_ops. A usage example is
   available in examples/filters/fblock_pick_variable.c

   See \ref BlockDataInterface for more details.
*/
void starpu_block_filter_pick_variable(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Return the child_ops of the partition obtained with starpu_block_filter_pick_variable().
   See \ref BlockDataInterface for more details.
*/
struct starpu_data_interface_ops *starpu_block_filter_pick_variable_child_ops(struct starpu_data_filter *f, unsigned child);

/** @} */

/**
   @name Predefined Tensor Filter Functions
   Predefined partitioning functions for tensor
   data.
   @{
*/

/**
  Partition a tensor along the X dimension, thus getting
  (x/\p nparts ,y,z,t) tensors. If \p nparts does not divide x, the last
  submatrix contains the remainder.

  See \ref TensorDataInterface for more details.
 */
void starpu_tensor_filter_block(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Partition a tensor along the X dimension, with a
   shadow border <c>filter_arg_ptr</c>, thus getting
   ((x-2*shadow)/\p nparts +2*shadow,y,z,t) tensors. If \p nparts does not
   divide x, the last submatrix contains the remainder.

   <b>IMPORTANT</b>:
   This can only be used for read-only access, as no coherency is
   enforced for the shadowed parts.

   See \ref TensorDataInterface for more details.
*/
void starpu_tensor_filter_block_shadow(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Partition a tensor along the Y dimension, thus getting
   (x,y/\p nparts ,z,t) tensors. If \p nparts does not divide y, the last
   submatrix contains the remainder.

   See \ref TensorDataInterface for more details.
 */
void starpu_tensor_filter_vertical_block(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Partition a tensor along the Y dimension, with a
   shadow border <c>filter_arg_ptr</c>, thus getting
   (x,(y-2*shadow)/\p nparts +2*shadow,z,t) tensors. If \p nparts does not
   divide y, the last submatrix contains the remainder.

   <b>IMPORTANT</b>:
   This can only be used for read-only access, as no coherency is
   enforced for the shadowed parts.

   See \ref TensorDataInterface for more details.
*/
void starpu_tensor_filter_vertical_block_shadow(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Partition a tensor along the Z dimension, thus getting
   (x,y,z/\p nparts,t) tensors. If \p nparts does not divide z, the last
   submatrix contains the remainder.

   See \ref TensorDataInterface for more details.
*/
void starpu_tensor_filter_depth_block(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Partition a tensor along the Z dimension, with a
   shadow border <c>filter_arg_ptr</c>, thus getting
   (x,y,(z-2*shadow)/\p nparts +2*shadow,t) tensors. If \p nparts does not
   divide z, the last submatrix contains the remainder.

   <b>IMPORTANT</b>:
   This can only be used for read-only access, as no coherency is
   enforced for the shadowed parts.

   See \ref TensorDataInterface for more details.
*/
void starpu_tensor_filter_depth_block_shadow(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Partition a tensor along the T dimension, thus getting
   (x,y,z,t/\p nparts) tensors. If \p nparts does not divide t, the last
   submatrix contains the remainder.

   See \ref TensorDataInterface for more details.
*/
void starpu_tensor_filter_time_block(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Partition a tensor along the T dimension, with a
   shadow border <c>filter_arg_ptr</c>, thus getting
   (x,y,z,(t-2*shadow)/\p nparts +2*shadow) tensors. If \p nparts does not
   divide t, the last submatrix contains the remainder.

   <b>IMPORTANT</b>:
   This can only be used for read-only access, as no coherency is
   enforced for the shadowed parts.

   See \ref TensorDataInterface for more details.
*/
void starpu_tensor_filter_time_block_shadow(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Pick \p nparts contiguous blocks from a tensor along
   the T dimension. The starting position on T-axis is set in
   <c>starpu_data_filter::filter_arg_ptr</c>.

   <c>starpu_data_filter::get_child_ops</c> needs to be set to
   starpu_tensor_filter_pick_block_child_ops. A usage example is
   available in examples/filters/ftensor_pick_block.c

   See \ref TensorDataInterface for more details.
*/
void starpu_tensor_filter_pick_block_t(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Pick \p nparts contiguous blocks from a tensor along
   the Z dimension. The starting position on Z-axis is set in
   <c>starpu_data_filter::filter_arg_ptr</c>.

   <c>starpu_data_filter::get_child_ops</c> needs to be set to
   starpu_tensor_filter_pick_block_child_ops. A usage example is
   available in examples/filters/ftensor_pick_block.c

   See \ref TensorDataInterface for more details.
*/
void starpu_tensor_filter_pick_block_z(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Pick \p nparts contiguous blocks from a tensor along
   the Y dimension. The starting position on Y-axis is set in
   <c>starpu_data_filter::filter_arg_ptr</c>.

   <c>starpu_data_filter::get_child_ops</c> needs to be set to
   starpu_tensor_filter_pick_block_child_ops. A usage example is
   available in examples/filters/ftensor_pick_block.c

   See \ref TensorDataInterface for more details.
*/
void starpu_tensor_filter_pick_block_y(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Return the child_ops of the partition obtained with starpu_tensor_filter_pick_block_t(),
   starpu_tensor_filter_pick_block_z() and starpu_tensor_filter_pick_block_y().
   See \ref TensorDataInterface for more details.
*/
struct starpu_data_interface_ops *starpu_tensor_filter_pick_block_child_ops(struct starpu_data_filter *f, unsigned child);

/**
   Pick \p nparts contiguous variables from a tensor. The starting position
   is set in <c>starpu_data_filter::filter_arg_ptr</c>.

   <c>starpu_data_filter::get_child_ops</c> needs to be set to
   starpu_tensor_filter_pick_variable_child_ops. A usage example is
   available in examples/filters/ftensor_pick_variable.c

   See \ref TensorDataInterface for more details.
*/
void starpu_tensor_filter_pick_variable(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Return the child_ops of the partition obtained with starpu_tensor_filter_pick_variable().
   See \ref TensorDataInterface for more details.
*/
struct starpu_data_interface_ops *starpu_tensor_filter_pick_variable_child_ops(struct starpu_data_filter *f, unsigned child);

/** @} */

/**
   @name Predefined Ndim Filter Functions
   Predefined partitioning functions for ndim array
   data.
   @{
*/

/**
   Partition a ndim array along the given dimension set in
   <c>starpu_data_filter::filter_arg</c>. If \p nparts does not
   divide the element number on dimension, the last submatrix contains the remainder.

   See \ref NdimDataInterface for more details.
 */
void starpu_ndim_filter_block(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Partition a ndim array along the given dimension set in
   <c>starpu_data_filter::filter_arg</c>, with a shadow border
   <c>starpu_data_filter::filter_arg_ptr</c>. If \p nparts does not
   divide the element number on dimension, the last submatrix contains the remainder.

   <b>IMPORTANT</b>:
   This can only be used for read-only access, as no coherency is
   enforced for the shadowed parts.

   See \ref NdimDataInterface for more details.
*/
void starpu_ndim_filter_block_shadow(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Partition a 4-dim array into \p nparts tensors along the given
   dimension set in <c>starpu_data_filter::filter_arg</c>.

   <c>starpu_data_filter::get_child_ops</c> needs to be set to
   starpu_ndim_filter_to_tensor_child_ops. A usage example is
   available in examples/filters/fndim_to_tensor.c

   See \ref NdimDataInterface for more details.
*/
void starpu_ndim_filter_to_tensor(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Partition a 3-dim array into \p nparts blocks along the given
   dimension set in <c>starpu_data_filter::filter_arg</c>.

   <c>starpu_data_filter::get_child_ops</c> needs to be set to
   starpu_ndim_filter_to_block_child_ops. A usage example is
   available in examples/filters/fndim_to_block.c

   See \ref NdimDataInterface for more details.
*/
void starpu_ndim_filter_to_block(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Partition a 2-dim array into \p nparts matrices along the given
   dimension set in <c>starpu_data_filter::filter_arg</c>.

   <c>starpu_data_filter::get_child_ops</c> needs to be set to
   starpu_ndim_filter_to_matrix_child_ops. A usage example is
   available in examples/filters/fndim_to_matrix.c

   See \ref NdimDataInterface for more details.
*/
void starpu_ndim_filter_to_matrix(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Partition a 1-dim array into \p nparts vectors.

   <c>starpu_data_filter::get_child_ops</c> needs to be set to
   starpu_ndim_filter_to_vector_child_ops. A usage example is
   available in examples/filters/fndim_to_vector.c

   See \ref NdimDataInterface for more details.
*/
void starpu_ndim_filter_to_vector(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Transfer a 0-dim array to a variable.

   <c>starpu_data_filter::get_child_ops</c> needs to be set to
   starpu_ndim_filter_to_variable_child_ops. A usage example is
   available in examples/filters/fndim_to_variable.c

   See \ref NdimDataInterface for more details.
*/
void starpu_ndim_filter_to_variable(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Pick \p nparts contiguous (n-1)dim arrays from a ndim array along
   the given dimension set in <c>starpu_data_filter::filter_arg</c>.
   The starting position is set in <c>starpu_data_filter::filter_arg_ptr</c>.

   A usage example is available in examples/filters/fndim_pick_ndim.c

   See \ref NdimDataInterface for more details.
*/
void starpu_ndim_filter_pick_ndim(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Pick \p nparts contiguous tensors from a 5-dim array along
   the given dimension set in <c>starpu_data_filter::filter_arg</c>.
   The starting position is set in <c>starpu_data_filter::filter_arg_ptr</c>.

   <c>starpu_data_filter::get_child_ops</c> needs to be set to
   starpu_ndim_filter_pick_tensor_child_ops. A usage example is
   available in examples/filters/fndim_5d_pick_tensor.c

   See \ref NdimDataInterface for more details.
*/
void starpu_ndim_filter_5d_pick_tensor(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Pick \p nparts contiguous blocks from a 4-dim array along
   the given dimension set in <c>starpu_data_filter::filter_arg</c>.
   The starting position is set in <c>starpu_data_filter::filter_arg_ptr</c>.

   <c>starpu_data_filter::get_child_ops</c> needs to be set to
   starpu_ndim_filter_pick_block_child_ops. A usage example is
   available in examples/filters/fndim_4d_pick_block.c

   See \ref NdimDataInterface for more details.
*/
void starpu_ndim_filter_4d_pick_block(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Pick \p nparts contiguous matrices from a 3-dim array along
   the given dimension set in <c>starpu_data_filter::filter_arg</c>.
   The starting position is set in <c>starpu_data_filter::filter_arg_ptr</c>.

   <c>starpu_data_filter::get_child_ops</c> needs to be set to
   starpu_ndim_filter_pick_matrix_child_ops. A usage example is
   available in examples/filters/fndim_3d_pick_matrix.c

   See \ref NdimDataInterface for more details.
*/
void starpu_ndim_filter_3d_pick_matrix(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Pick \p nparts contiguous vectors from a 2-dim array along
   the given dimension set in <c>starpu_data_filter::filter_arg</c>.
   The starting position is set in <c>starpu_data_filter::filter_arg_ptr</c>.

   <c>starpu_data_filter::get_child_ops</c> needs to be set to
   starpu_ndim_filter_pick_vector_child_ops. A usage example is
   available in examples/filters/fndim_2d_pick_vector.c

   See \ref NdimDataInterface for more details.
*/
void starpu_ndim_filter_2d_pick_vector(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Pick \p nparts contiguous variables from a 1-dim array.
   The starting position is set in <c>starpu_data_filter::filter_arg_ptr</c>.

   <c>starpu_data_filter::get_child_ops</c> needs to be set to
   starpu_ndim_filter_pick_variable_child_ops. A usage example is
   available in examples/filters/fndim_1d_pick_variable.c

   See \ref NdimDataInterface for more details.
*/
void starpu_ndim_filter_1d_pick_variable(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Pick \p nparts contiguous variables from a ndim array.
   The starting position is set in <c>starpu_data_filter::filter_arg_ptr</c>.

   <c>starpu_data_filter::get_child_ops</c> needs to be set to
   starpu_ndim_filter_pick_variable_child_ops. A usage example is
   available in examples/filters/fndim_pick_variable.c

   See \ref NdimDataInterface for more details.
*/
void starpu_ndim_filter_pick_variable(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);

/**
   Return the child_ops of the partition obtained with starpu_ndim_filter_pick_tensor().
   See \ref NdimDataInterface for more details.
*/
struct starpu_data_interface_ops *starpu_ndim_filter_pick_tensor_child_ops(struct starpu_data_filter *f, unsigned child);

/**
   Return the child_ops of the partition obtained with starpu_ndim_filter_pick_block().
   See \ref NdimDataInterface for more details.
*/
struct starpu_data_interface_ops *starpu_ndim_filter_pick_block_child_ops(struct starpu_data_filter *f, unsigned child);

/**
   Return the child_ops of the partition obtained with starpu_ndim_filter_pick_matrix().
   See \ref NdimDataInterface for more details.
*/
struct starpu_data_interface_ops *starpu_ndim_filter_pick_matrix_child_ops(struct starpu_data_filter *f, unsigned child);

/**
   Return the child_ops of the partition obtained with starpu_ndim_filter_pick_vector().
   See \ref NdimDataInterface for more details.
*/
struct starpu_data_interface_ops *starpu_ndim_filter_pick_vector_child_ops(struct starpu_data_filter *f, unsigned child);

/**
   Return the child_ops of the partition obtained with starpu_ndim_filter_pick_variable().
   See \ref NdimDataInterface for more details.
*/
struct starpu_data_interface_ops *starpu_ndim_filter_pick_variable_child_ops(struct starpu_data_filter *f, unsigned child);

/**
   Return the child_ops of the partition obtained with starpu_ndim_filter_to_tensor().
   See \ref NdimDataInterface for more details.
*/
struct starpu_data_interface_ops *starpu_ndim_filter_to_tensor_child_ops(struct starpu_data_filter *f, unsigned child);

/**
   Return the child_ops of the partition obtained with starpu_ndim_filter_to_block().
   See \ref NdimDataInterface for more details.
*/
struct starpu_data_interface_ops *starpu_ndim_filter_to_block_child_ops(struct starpu_data_filter *f, unsigned child);

/**
   Return the child_ops of the partition obtained with starpu_ndim_filter_to_matrix().
   See \ref NdimDataInterface for more details.
*/
struct starpu_data_interface_ops *starpu_ndim_filter_to_matrix_child_ops(struct starpu_data_filter *f, unsigned child);

/**
   Return the child_ops of the partition obtained with starpu_ndim_filter_to_vector().
   See \ref NdimDataInterface for more details.
*/
struct starpu_data_interface_ops *starpu_ndim_filter_to_vector_child_ops(struct starpu_data_filter *f, unsigned child);

/**
   Return the child_ops of the partition obtained with starpu_ndim_filter_to_variable().
   See \ref NdimDataInterface for more details.
*/
struct starpu_data_interface_ops *starpu_ndim_filter_to_variable_child_ops(struct starpu_data_filter *f, unsigned child);

/**
   Given an integer \p n, \p n the number of parts it must be divided in, \p id the
   part currently considered, determines the \p chunk_size and the \p offset, taking
   into account the size of the elements stored in the data structure \p elemsize
   and \p blocksize, which is most often 1.
   See \ref DefiningANewDataFilter for more details.
 */
void starpu_filter_nparts_compute_chunk_size_and_offset(unsigned n, unsigned nparts,
							size_t elemsize, unsigned id,
							unsigned blocksize, unsigned *chunk_size,
							size_t *offset);

/** @} */

/** @} */

#ifdef __cplusplus
}
#endif

#endif
