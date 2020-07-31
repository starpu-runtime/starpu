/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2011       Télécom-SudParis
 * Copyright (C) 2016       Uppsala University
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

#ifndef __STARPU_TASK_H__
#define __STARPU_TASK_H__

#include <starpu.h>
#include <errno.h>
#include <assert.h>

#if defined STARPU_USE_CUDA && !defined STARPU_DONT_INCLUDE_CUDA_HEADERS
# include <cuda.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

/**
   @defgroup API_Codelet_And_Tasks Codelet And Tasks
   @brief This section describes the interface to manipulate codelets
   and tasks.
   @{
*/

/**
   To be used when setting the field starpu_codelet::where to specify
   that the codelet has no computation part, and thus does not need to
   be scheduled, and data does not need to be actually loaded. This is
   thus essentially used for synchronization tasks.
*/
#define STARPU_NOWHERE	((1ULL)<<0)

/**
   To be used when setting the field starpu_codelet::where (or
   starpu_task::where) to specify the codelet (or the task) may be
   executed on a CPU processing unit.
*/
#define STARPU_CPU	((1ULL)<<1)

/**
   To be used when setting the field starpu_codelet::where (or
   starpu_task::where) to specify the codelet (or the task) may be
   executed on a CUDA processing unit.
*/
#define STARPU_CUDA	((1ULL)<<3)

/**
   To be used when setting the field starpu_codelet::where (or
   starpu_task::where) to specify the codelet (or the task) may be
   executed on a OpenCL processing unit.
*/
#define STARPU_OPENCL	((1ULL)<<6)

/**
   To be used when setting the field starpu_codelet::where (or
   starpu_task::where) to specify the codelet (or the task) may be
   executed on a MIC processing unit.
*/
#define STARPU_MIC	((1ULL)<<7)

/**
   To be used when setting the field starpu_codelet::where (or
   starpu_task::where) to specify the codelet (or the task) may be
   executed on a MPI Slave processing unit.
*/
#define STARPU_MPI_MS	((1ULL)<<9)

/**
   Value to be set in starpu_codelet::flags to execute the codelet
   functions even in simgrid mode.
*/
#define STARPU_CODELET_SIMGRID_EXECUTE	(1<<0)

/**
   Value to be set in starpu_codelet::flags to execute the codelet
   functions even in simgrid mode, and later inject the measured
   timing inside the simulation.
*/
#define STARPU_CODELET_SIMGRID_EXECUTE_AND_INJECT	(1<<1)

/**
   Value to be set in starpu_codelet::flags to make starpu_task_submit()
   not submit automatic asynchronous partitioning/unpartitioning.
*/
#define STARPU_CODELET_NOPLANS	(1<<2)

/**
   Value to be set in starpu_codelet::cuda_flags to allow asynchronous
   CUDA kernel execution.
 */
#define STARPU_CUDA_ASYNC	(1<<0)

/**
   Value to be set in starpu_codelet::opencl_flags to allow
   asynchronous OpenCL kernel execution.
*/
#define STARPU_OPENCL_ASYNC	(1<<0)

/**
   To be used when the RAM memory node is specified.
*/
#define STARPU_MAIN_RAM 0


/**
   Describe the type of parallel task. See \ref ParallelTasks for
   details.
*/
enum starpu_codelet_type
{
	STARPU_SEQ = 0,  /**< (default) for classical sequential
			    tasks.
			 */
	STARPU_SPMD,     /**< for a parallel task whose threads are
			    handled by StarPU, the code has to use
			    starpu_combined_worker_get_size() and
			    starpu_combined_worker_get_rank() to
			    distribute the work.
			 */
	STARPU_FORKJOIN /**< for a parallel task whose threads are
			   started by the codelet function, which has
			   to use starpu_combined_worker_get_size() to
			   determine how many threads should be
			   started.
			*/
};

enum starpu_task_status
{
	STARPU_TASK_INIT,        /**< The task has just been initialized. */
#define STARPU_TASK_INIT 0
#define STARPU_TASK_INVALID STARPU_TASK_INIT  /**< old name for STARPU_TASK_INIT */
	STARPU_TASK_BLOCKED,     /**< The task has just been
				    submitted, and its dependencies has not been checked yet. */
	STARPU_TASK_READY,       /**< The task is ready for execution. */
	STARPU_TASK_RUNNING,     /**< The task is running on some worker. */
	STARPU_TASK_FINISHED,    /**< The task is finished executing. */
	STARPU_TASK_BLOCKED_ON_TAG,  /**< The task is waiting for a tag. */
	STARPU_TASK_BLOCKED_ON_TASK, /**< The task is waiting for a task. */
	STARPU_TASK_BLOCKED_ON_DATA, /**< The task is waiting for some data. */
	STARPU_TASK_STOPPED          /**< The task is stopped. */
};

/**
   CPU implementation of a codelet.
*/
typedef void (*starpu_cpu_func_t)(void **, void*);

/**
   CUDA implementation of a codelet.
*/
typedef void (*starpu_cuda_func_t)(void **, void*);

/**
   OpenCL implementation of a codelet.
*/
typedef void (*starpu_opencl_func_t)(void **, void*);

/**
   MIC implementation of a codelet.
*/
typedef void (*starpu_mic_kernel_t)(void **, void*);

/**
  MIC kernel for a codelet
*/
typedef starpu_mic_kernel_t (*starpu_mic_func_t)(void);

/**
   MPI Master Slave kernel for a codelet
*/
typedef void (*starpu_mpi_ms_kernel_t)(void **, void*);

/**
   MPI Master Slave implementation of a codelet.
*/
typedef starpu_mpi_ms_kernel_t (*starpu_mpi_ms_func_t)(void);

/**
   @deprecated
   Setting the field starpu_codelet::cpu_func with this macro
   indicates the codelet will have several implementations. The use of
   this macro is deprecated. One should always only define the field
   starpu_codelet::cpu_funcs.
*/
#define STARPU_MULTIPLE_CPU_IMPLEMENTATIONS    ((starpu_cpu_func_t) -1)

/**
   @deprecated
   Setting the field starpu_codelet::cuda_func with this macro
   indicates the codelet will have several implementations. The use of
   this macro is deprecated. One should always only define the field
   starpu_codelet::cuda_funcs.
*/
#define STARPU_MULTIPLE_CUDA_IMPLEMENTATIONS   ((starpu_cuda_func_t) -1)

/**
   @deprecated
   Setting the field starpu_codelet::opencl_func with this macro
   indicates the codelet will have several implementations. The use of
   this macro is deprecated. One should always only define the field
   starpu_codelet::opencl_funcs.
*/
#define STARPU_MULTIPLE_OPENCL_IMPLEMENTATIONS ((starpu_opencl_func_t) -1)

/**
   Value to set in starpu_codelet::nbuffers to specify that the
   codelet can accept a variable number of buffers, specified in
   starpu_task::nbuffers.
*/
#define STARPU_VARIABLE_NBUFFERS (-1)

/**
   Value to be set in the field starpu_codelet::nodes to request
   StarPU to put the data in CPU-accessible memory (and let StarPU
   choose the NUMA node).
*/
#define STARPU_SPECIFIC_NODE_LOCAL (-1)
#define STARPU_SPECIFIC_NODE_CPU (-2)
#define STARPU_SPECIFIC_NODE_SLOW (-3)
#define STARPU_SPECIFIC_NODE_FAST (-4)

struct starpu_task;

/**
   The codelet structure describes a kernel that is possibly
   implemented on various targets. For compatibility, make sure to
   initialize the whole structure to zero, either by using explicit
   memset, or the function starpu_codelet_init(), or by letting the
   compiler implicitly do it in e.g. static storage case.
*/
struct starpu_codelet
{
	/**
	   Optional field to indicate which types of processing units
	   are able to execute the codelet. The different values
	   ::STARPU_CPU, ::STARPU_CUDA, ::STARPU_OPENCL can be
	   combined to specify on which types of processing units the
	   codelet can be executed. ::STARPU_CPU|::STARPU_CUDA for
	   instance indicates that the codelet is implemented for both
	   CPU cores and CUDA devices while ::STARPU_OPENCL indicates
	   that it is only available on OpenCL devices. If the field
	   is unset, its value will be automatically set based on the
	   availability of the XXX_funcs fields defined below. It can
	   also be set to ::STARPU_NOWHERE to specify that no
	   computation has to be actually done.
	*/
	uint32_t where;

	/**
	   Define a function which should return 1 if the worker
	   designated by \p workerid can execute the \p nimpl -th
	   implementation of \p task, 0 otherwise.
	*/
	int (*can_execute)(unsigned workerid, struct starpu_task *task, unsigned nimpl);

	/**
	   Optional field to specify the type of the codelet. The
	   default is ::STARPU_SEQ, i.e. usual sequential
	   implementation. Other values (::STARPU_SPMD or
	   ::STARPU_FORKJOIN) declare that a parallel implementation is
	   also available. See \ref ParallelTasks for details.
	*/
	enum starpu_codelet_type type;

	/**
	   Optional field. If a parallel implementation is available,
	   this denotes the maximum combined worker size that StarPU
	   will use to execute parallel tasks for this codelet.
	*/
	int max_parallelism;

	/**
	   @deprecated
	   Optional field which has been made deprecated. One should
	   use instead the field starpu_codelet::cpu_funcs.
	*/
	starpu_cpu_func_t cpu_func STARPU_DEPRECATED;

	/**
	   @deprecated
	   Optional field which has been made deprecated. One should
	   use instead the starpu_codelet::cuda_funcs field.
	*/
	starpu_cuda_func_t cuda_func STARPU_DEPRECATED;

	/**
	   @deprecated
	   Optional field which has been made deprecated. One should
	   use instead the starpu_codelet::opencl_funcs field.
	*/
	starpu_opencl_func_t opencl_func STARPU_DEPRECATED;

	/**
	   Optional array of function pointers to the CPU
	   implementations of the codelet. The functions prototype
	   must be:
	   \code{.c}
	   void cpu_func(void *buffers[], void *cl_arg)
	   \endcode
	   The first argument being the array of data managed by the
	   data management library, and the second argument is a
	   pointer to the argument passed from the field
	   starpu_task::cl_arg. If the field starpu_codelet::where is
	   set, then the field tarpu_codelet::cpu_funcs is ignored if
	   ::STARPU_CPU does not appear in the field
	   starpu_codelet::where, it must be non-<c>NULL</c> otherwise.
	*/
	starpu_cpu_func_t cpu_funcs[STARPU_MAXIMPLEMENTATIONS];

	/**
	   Optional array of function pointers to the CUDA
	   implementations of the codelet. The functions must be
	   host-functions written in the CUDA runtime API. Their
	   prototype must be:
	   \code{.c}
	   void cuda_func(void *buffers[], void *cl_arg)
	   \endcode
	   If the field starpu_codelet::where is set, then the field
	   starpu_codelet::cuda_funcs is ignored if ::STARPU_CUDA does
	   not appear in the field starpu_codelet::where, it must be
	   non-<c>NULL</c> otherwise.
	*/
	starpu_cuda_func_t cuda_funcs[STARPU_MAXIMPLEMENTATIONS];

	/**
	   Optional array of flags for CUDA execution. They specify
	   some semantic details about CUDA kernel execution, such as
	   asynchronous execution.
	*/
	char cuda_flags[STARPU_MAXIMPLEMENTATIONS];

	/**
	   Optional array of function pointers to the OpenCL
	   implementations of the codelet. The functions prototype
	   must be:
	   \code{.c}
	   void opencl_func(void *buffers[], void *cl_arg)
	   \endcode
	   If the field starpu_codelet::where field is set, then the
	   field starpu_codelet::opencl_funcs is ignored if
	   ::STARPU_OPENCL does not appear in the field
	   starpu_codelet::where, it must be non-<c>NULL</c> otherwise.
	*/
	starpu_opencl_func_t opencl_funcs[STARPU_MAXIMPLEMENTATIONS];

	/**
	   Optional array of flags for OpenCL execution. They specify
	   some semantic details about OpenCL kernel execution, such
	   as asynchronous execution.
	*/
	char opencl_flags[STARPU_MAXIMPLEMENTATIONS];

	/**
	   Optional array of function pointers to a function which
	   returns the MIC implementation of the codelet. The
	   functions prototype must be:
	   \code{.c}
	   starpu_mic_kernel_t mic_func(struct starpu_codelet *cl, unsigned nimpl)
	   \endcode
	   If the field starpu_codelet::where is set, then the field
	   starpu_codelet::mic_funcs is ignored if ::STARPU_MIC does
	   not appear in the field starpu_codelet::where. It can be
	   <c>NULL</c> if starpu_codelet::cpu_funcs_name is
	   non-<c>NULL</c>, in which case StarPU will simply make a
	   symbol lookup to get the implementation.
	*/
	starpu_mic_func_t mic_funcs[STARPU_MAXIMPLEMENTATIONS];

	/**
	   Optional array of function pointers to a function which
	   returns the MPI Master Slave implementation of the codelet.
	   The functions prototype must be:
	   \code{.c}
	   starpu_mpi_ms_kernel_t mpi_ms_func(struct starpu_codelet *cl, unsigned nimpl)
	   \endcode
	   If the field starpu_codelet::where is set, then the field
	   starpu_codelet::mpi_ms_funcs is ignored if ::STARPU_MPI_MS
	   does not appear in the field starpu_codelet::where. It can
	   be <c>NULL</c> if starpu_codelet::cpu_funcs_name is
	   non-<c>NULL</c>, in which case StarPU will simply make a
	   symbol lookup to get the implementation.
	*/
	starpu_mpi_ms_func_t mpi_ms_funcs[STARPU_MAXIMPLEMENTATIONS];

	/**
	   Optional array of strings which provide the name of the CPU
	   functions referenced in the array
	   starpu_codelet::cpu_funcs. This can be used when running on
	   MIC devices for StarPU to simply look
	   up the MIC function implementation through its name.
	*/
	const char *cpu_funcs_name[STARPU_MAXIMPLEMENTATIONS];

	/**
	   Specify the number of arguments taken by the codelet. These
	   arguments are managed by the DSM and are accessed from the
	   <c>void *buffers[]</c> array. The constant argument passed
	   with the field starpu_task::cl_arg is not counted in this
	   number. This value should not be above \ref
	   STARPU_NMAXBUFS. It may be set to \ref
	   STARPU_VARIABLE_NBUFFERS to specify that the number of
	   buffers and their access modes will be set in
	   starpu_task::nbuffers and starpu_task::modes or
	   starpu_task::dyn_modes, which thus permits to define
	   codelets with a varying number of data.
	*/
	int nbuffers;

	/**
	   Is an array of ::starpu_data_access_mode. It describes the
	   required access modes to the data neeeded by the codelet
	   (e.g. ::STARPU_RW). The number of entries in this array
	   must be specified in the field starpu_codelet::nbuffers,
	   and should not exceed \ref STARPU_NMAXBUFS. If
	   unsufficient, this value can be set with the configure
	   option \ref enable-maxbuffers "--enable-maxbuffers".
	*/
	enum starpu_data_access_mode modes[STARPU_NMAXBUFS];

	/**
	   Is an array of ::starpu_data_access_mode. It describes the
	   required access modes to the data needed by the codelet
	   (e.g. ::STARPU_RW). The number of entries in this array
	   must be specified in the field starpu_codelet::nbuffers.
	   This field should be used for codelets having a number of
	   datas greater than \ref STARPU_NMAXBUFS (see \ref
	   SettingManyDataHandlesForATask). When defining a codelet,
	   one should either define this field or the field
	   starpu_codelet::modes defined above.
	*/
	enum starpu_data_access_mode *dyn_modes;

	/**
	   Default value is 0. If this flag is set, StarPU will not
	   systematically send all data to the memory node where the
	   task will be executing, it will read the
	   starpu_codelet::nodes or starpu_codelet::dyn_nodes array to
	   determine, for each data, whether to send it on the memory
	   node where the task will be executing (-1), or on a
	   specific node (!= -1).
	*/
	unsigned specific_nodes;

	/**
	   Optional field. When starpu_codelet::specific_nodes is 1,
	   this specifies the memory nodes where each data should be
	   sent to for task execution. The number of entries in this
	   array is starpu_codelet::nbuffers, and should not exceed
	   \ref STARPU_NMAXBUFS.
	*/
	int nodes[STARPU_NMAXBUFS];

	/**
	   Optional field. When starpu_codelet::specific_nodes is 1,
	   this specifies the memory nodes where each data should be
	   sent to for task execution. The number of entries in this
	   array is starpu_codelet::nbuffers. This field should be
	   used for codelets having a number of datas greater than
	   \ref STARPU_NMAXBUFS (see \ref
	   SettingManyDataHandlesForATask). When defining a codelet,
	   one should either define this field or the field
	   starpu_codelet::nodes defined above.
	*/
	int *dyn_nodes;

	/**
	   Optional pointer to the task duration performance model
	   associated to this codelet. This optional field is ignored
	   when set to <c>NULL</c> or when its field
	   starpu_perfmodel::symbol is not set.
	*/
	struct starpu_perfmodel *model;

	/**
	   Optional pointer to the task energy consumption performance
	   model associated to this codelet. This optional field is
	   ignored when set to <c>NULL</c> or when its field
	   starpu_perfmodel::symbol is not set. In the case of
	   parallel codelets, this has to account for all processing
	   units involved in the parallel execution.
	*/
	struct starpu_perfmodel *energy_model;

	/**
	   Optional array for statistics collected at runtime: this is
	   filled by StarPU and should not be accessed directly, but
	   for example by calling the function
	   starpu_codelet_display_stats() (See
	   starpu_codelet_display_stats() for details).
	 */
	unsigned long per_worker_stats[STARPU_NMAXWORKERS];

	/**
	   Optional name of the codelet. This can be useful for
	   debugging purposes.
	 */
	const char *name;

	/**
	   Optional color of the codelet. This can be useful for
	   debugging purposes.
	*/
	unsigned color;

	/**
	   Various flags for the codelet.
	 */
	int flags;

	/**
	   Whether _starpu_codelet_check_deprecated_fields was already done or not.
	 */
	int checked;
};

/**
   Describe a data handle along with an access mode.
*/
struct starpu_data_descr
{
	starpu_data_handle_t handle; /**< data */
	enum starpu_data_access_mode mode; /**< access mode */
};

/**
   Describe a task that can be offloaded on the various processing
   units managed by StarPU. It instantiates a codelet. It can either
   be allocated dynamically with the function starpu_task_create(), or
   declared statically. In the latter case, the programmer has to zero
   the structure starpu_task and to fill the different fields
   properly. The indicated default values correspond to the
   configuration of a task allocated with starpu_task_create().
*/
struct starpu_task
{
	/**
	   Optional name of the task. This can be useful for debugging
	   purposes.

	   With starpu_task_insert() and alike this can be specified thanks to
	   ::STARPU_NAME followed by the const char *.
	*/
	const char *name;

	/**
	   Pointer to the corresponding structure starpu_codelet. This
	   describes where the kernel should be executed, and supplies
	   the appropriate implementations. When set to <c>NULL</c>,
	   no code is executed during the tasks, such empty tasks can
	   be useful for  synchronization purposes.
	*/
	struct starpu_codelet *cl;

	/**
	   When set, specify where the task is allowed to be executed.
	   When unset, take the value of starpu_codelet::where.

	   With starpu_task_insert() and alike this can be specified thanks to
	   ::STARPU_EXECUTE_WHERE followed by an unsigned long long.
	*/
	int32_t where;

	/**
	   Specify the number of buffers. This is only used when
	   starpu_codelet::nbuffers is \ref STARPU_VARIABLE_NBUFFERS.

	   With starpu_task_insert() and alike this is automatically computed
	   when using ::STARPU_DATA_ARRAY and alike.
	*/
	int nbuffers;

        /* Keep dyn_handles, dyn_interfaces and dyn_modes before the
	 * equivalent static arrays, so we can detect dyn_handles
	 * being NULL while nbuffers being bigger that STARPU_NMAXBUFS
	 * (otherwise the overflow would put a non-NULL) */

	/**
	   Array of ::starpu_data_handle_t. Specify the handles to the
	   different pieces of data accessed by the task. The number
	   of entries in this array must be specified in the field
	   starpu_codelet::nbuffers. This field should be used for
	   tasks having a number of datas greater than \ref
	   STARPU_NMAXBUFS (see \ref SettingManyDataHandlesForATask).
	   When defining a task, one should either define this field
	   or the field starpu_task::handles defined below.

	   With starpu_task_insert() and alike this is automatically filled
	   when using ::STARPU_DATA_ARRAY and alike.
	*/
	starpu_data_handle_t *dyn_handles;
	/**
	   Array of data pointers to the memory node where execution
	   will happen, managed by the DSM. Is used when the field
	   starpu_task::dyn_handles is defined.

	   This is filled by StarPU.
	*/
	void **dyn_interfaces;
	/**
	   Used only when starpu_codelet::nbuffers is \ref
	   STARPU_VARIABLE_NBUFFERS.
	   Array of ::starpu_data_access_mode which describes the
	   required access modes to the data needed by the codelet
	   (e.g. ::STARPU_RW). The number of entries in this array
	   must be specified in the field starpu_codelet::nbuffers.
	   This field should be used for codelets having a number of
	   datas greater than \ref STARPU_NMAXBUFS (see \ref
	   SettingManyDataHandlesForATask).
	   When defining a codelet, one should either define this
	   field or the field starpu_task::modes defined below.

	   With starpu_task_insert() and alike this is automatically filled
	   when using ::STARPU_DATA_MODE_ARRAY and alike.
	*/
	enum starpu_data_access_mode *dyn_modes;

	/**
	   Array of ::starpu_data_handle_t. Specify the handles to the
	   different pieces of data accessed by the task. The number
	   of entries in this array must be specified in the field
	   starpu_codelet::nbuffers, and should not exceed
	   \ref STARPU_NMAXBUFS. If unsufficient, this value can be
	   set with the configure option \ref enable-maxbuffers
	   "--enable-maxbuffers".

	   With starpu_task_insert() and alike this is automatically filled
	   when using ::STARPU_R and alike.
	*/
	starpu_data_handle_t handles[STARPU_NMAXBUFS];
	/**
	   Array of Data pointers to the memory node where execution
	   will happen, managed by the DSM.

	   This is filled by StarPU.
	*/
	void *interfaces[STARPU_NMAXBUFS];
	/**
	   Used only when starpu_codelet::nbuffers is \ref
	   STARPU_VARIABLE_NBUFFERS.
	   Array of ::starpu_data_access_mode which describes the
	   required access modes to the data neeeded by the codelet
	   (e.g. ::STARPU_RW). The number of entries in this array
	   must be specified in the field starpu_task::nbuffers, and
	   should not exceed \ref STARPU_NMAXBUFS. If unsufficient,
	   this value can be set with the configure option
	   \ref enable-maxbuffers "--enable-maxbuffers".

	   With starpu_task_insert() and alike this is automatically filled
	   when using ::STARPU_DATA_MODE_ARRAY and alike.
	*/
	enum starpu_data_access_mode modes[STARPU_NMAXBUFS];

	/**
	   Optional pointer to an array of characters which allows to
	   define the sequential consistency for each handle for the
	   current task.

	   With starpu_task_insert() and alike this can be specified thanks to
	   ::STARPU_HANDLES_SEQUENTIAL_CONSISTENCY followed by an unsigned char *
	*/
	unsigned char *handles_sequential_consistency;

	/**
	   Optional pointer which is passed to the codelet through the
	   second argument of the codelet implementation (e.g.
	   starpu_codelet::cpu_func or starpu_codelet::cuda_func). The
	   default value is <c>NULL</c>. starpu_codelet_pack_args()
	   and starpu_codelet_unpack_args() are helpers that can can
	   be used to respectively pack and unpack data into and from
	   it, but the application can manage it any way, the only
	   requirement is that the size of the data must be set in
	   starpu_task::cl_arg_size .

	   With starpu_task_insert() and alike this can be specified thanks to
	   ::STARPU_CL_ARGS followed by a void* and a size_t.
	*/
	void *cl_arg;
	/**
	   Optional field. For some specific drivers, the pointer
	   starpu_task::cl_arg cannot not be directly given to the
	   driver function. A buffer of size starpu_task::cl_arg_size
	   needs to be allocated on the driver. This buffer is then
	   filled with the starpu_task::cl_arg_size bytes starting at
	   address starpu_task::cl_arg. In this case, the argument
	   given to the codelet is therefore not the
	   starpu_task::cl_arg pointer, but the address of the buffer
	   in local store (LS) instead. This field is ignored for CPU,
	   CUDA and OpenCL codelets, where the starpu_task::cl_arg
	   pointer is given as such.

	   With starpu_task_insert() and alike this can be specified thanks to
	   ::STARPU_CL_ARGS followed by a void* and a size_t.
	*/
	size_t cl_arg_size;

	/**
	   Optional field, the default value is <c>NULL</c>. This is a
	   function pointer of prototype <c>void (*f)(void *)</c>
	   which specifies a possible callback. If this pointer is
	   non-<c>NULL</c>, the callback function is executed on the
	   host after the execution of the task. Tasks which depend on
	   it might already be executing. The callback is passed the
	   value contained in the starpu_task::callback_arg field. No
	   callback is executed if the field is set to <c>NULL</c>.

	   With starpu_task_insert() and alike this can be specified thanks to
	   ::STARPU_CALLBACK followed by the function pointer, or thanks to
	   ::STARPU_CALLBACK_WITH_ARG (or
	   ::STARPU_CALLBACK_WITH_ARG_NFREE) followed by the function
	   pointer and the argument.
	*/
	void (*callback_func)(void *);
	/**
	   Optional field, the default value is <c>NULL</c>. This is
	   the pointer passed to the callback function. This field is
	   ignored if the field starpu_task::callback_func is set to
	   <c>NULL</c>.

	   With starpu_task_insert() and alike this can be specified thanks to
	   ::STARPU_CALLBACK_ARG followed by the function pointer, or thanks to
	   ::STARPU_CALLBACK_WITH_ARG or
	   ::STARPU_CALLBACK_WITH_ARG_NFREE followed by the function
	   pointer and the argument.
	*/
	void *callback_arg;

	/**
	   Optional field, the default value is <c>NULL</c>. This is a
	   function pointer of prototype <c>void (*f)(void *)</c>
	   which specifies a possible callback. If this pointer is
	   non-<c>NULL</c>, the callback function is executed on the
	   host when the task becomes ready for execution, before
	   getting scheduled. The callback is passed the value
	   contained in the starpu_task::prologue_callback_arg field.
	   No callback is executed if the field is set to <c>NULL</c>.

	   With starpu_task_insert() and alike this can be specified thanks to
	   ::STARPU_PROLOGUE_CALLBACK followed by the function pointer.
	*/
	void (*prologue_callback_func)(void *);

	/**
	   Optional field, the default value is <c>NULL</c>. This is
	   the pointer passed to the prologue callback function. This
	   field is ignored if the field
	   starpu_task::prologue_callback_func is set to <c>NULL</c>.

	   With starpu_task_insert() and alike this can be specified thanks to
	   ::STARPU_PROLOGUE_CALLBACK_ARG followed by the argument
	*/
	void *prologue_callback_arg;

	void (*prologue_callback_pop_func)(void *);
	void *prologue_callback_pop_arg;

	/**
	    Optional field. Contain the tag associated to the task if
	    the field starpu_task::use_tag is set, ignored
	    otherwise.

	   With starpu_task_insert() and alike this can be specified thanks to
	   ::STARPU_TAG followed by a starpu_tag_t.
	*/
	starpu_tag_t tag_id;

	/**
	   Optional field. In case starpu_task::cl_arg was allocated
	   by the application through <c>malloc()</c>, setting
	   starpu_task::cl_arg_free to 1 makes StarPU automatically
	   call <c>free(cl_arg)</c> when destroying the task. This
	   saves the user from defining a callback just for that. This
	   is mostly useful when targetting MIC, where the
	   codelet does not execute in the same memory space as the
	   main thread.

	   With starpu_task_insert() and alike this is set to 1 when using
	   ::STARPU_CL_ARGS.
	*/
	unsigned cl_arg_free:1;

	/**
	   Optional field. In case starpu_task::callback_arg was
	   allocated by the application through <c>malloc()</c>,
	   setting starpu_task::callback_arg_free to 1 makes StarPU
	   automatically call <c>free(callback_arg)</c> when
	   destroying the task.

	   With starpu_task_insert() and alike, this is set to 1 when using
	   ::STARPU_CALLBACK_ARG or ::STARPU_CALLBACK_WITH_ARG, or set
	   to 0 when using ::STARPU_CALLBACK_ARG_NFREE
	*/
	unsigned callback_arg_free:1;

	/**
	   Optional field. In case starpu_task::prologue_callback_arg
	   was allocated by the application through <c>malloc()</c>,
	   setting starpu_task::prologue_callback_arg_free to 1 makes
	   StarPU automatically call
	   <c>free(prologue_callback_arg)</c> when destroying the task.

	   With starpu_task_insert() and alike this is set to 1 when using
	   ::STARPU_PROLOGUE_CALLBACK_ARG, or set to 0 when using
	   ::STARPU_PROLOGUE_CALLBACK_ARG_NFREE
	*/
	unsigned prologue_callback_arg_free:1;

	/**
	   Optional field. In case starpu_task::prologue_callback_pop_arg
	   was allocated by the application through <c>malloc()</c>,
	   setting starpu_task::prologue_callback_pop_arg_free to 1 makes
	   StarPU automatically call
	   <c>free(prologue_callback_pop_arg)</c> when destroying the
	   task.

	   With starpu_task_insert() and alike this is set to 1 when using
	   ::STARPU_PROLOGUE_CALLBACK_POP_ARG, or set to 0 when using
	   ::STARPU_PROLOGUE_CALLBACK_POP_ARG_NFREE
	*/
	unsigned prologue_callback_pop_arg_free:1;

	/**
	   Optional field, the default value is 0. If set, this flag
	   indicates that the task should be associated with the tag
	   contained in the starpu_task::tag_id field. Tag allow the
	   application to synchronize with the task and to express
	   task dependencies easily.

	   With starpu_task_insert() and alike this is set to 1 when using
	   ::STARPU_TAG.
	*/
	unsigned use_tag:1;

	/**
	   If this flag is set (which is the default), sequential
	   consistency is enforced for the data parameters of this
	   task for which sequential consistency is enabled. Clearing
	   this flag permits to disable sequential consistency for
	   this task, even if data have it enabled.

	   With starpu_task_insert() and alike this can be specified thanks to
	   ::STARPU_SEQUENTIAL_CONSISTENCY followed by an unsigned.
	*/
	unsigned sequential_consistency:1;

	/**
	   If this flag is set, the function starpu_task_submit() is
	   blocking and returns only when the task has been executed
	   (or if no worker is able to process the task). Otherwise,
	   starpu_task_submit() returns immediately.

	   With starpu_task_insert() and alike this can be specified thanks to
	   ::STARPU_TASK_SYNCHRONOUS followed an int.
	*/
	unsigned synchronous:1;

	/**
	   Default value is 0. If this flag is set, StarPU will bypass
	   the scheduler and directly affect this task to the worker
	   specified by the field starpu_task::workerid.

	   With starpu_task_insert() and alike this is set to 1 when using
	   ::STARPU_EXECUTE_ON_WORKER.
	*/
	unsigned execute_on_a_specific_worker:1;

	/**
	   Optional field, default value is 1. If this flag is set, it
	   is not possible to synchronize with the task by the means
	   of starpu_task_wait() later on. Internal data structures
	   are only guaranteed to be freed once starpu_task_wait() is
	   called if the flag is not set.

	   With starpu_task_insert() and alike this is set to 1.
	*/
	unsigned detach:1;

	/**
	   Optional value. Default value is 0 for starpu_task_init(),
	   and 1 for starpu_task_create(). If this flag is set, the
	   task structure will automatically be freed, either after
	   the execution of the callback if the task is detached, or
	   during starpu_task_wait() otherwise. If this flag is not
	   set, dynamically allocated data structures will not be
	   freed until starpu_task_destroy() is called explicitly.
	   Setting this flag for a statically allocated task structure
	   will result in undefined behaviour. The flag is set to 1
	   when the task is created by calling starpu_task_create().
	   Note that starpu_task_wait_for_all() will not free any task.

	   With starpu_task_insert() and alike this is set to 1.
	*/
	unsigned destroy:1;

	/**
	   Optional field. If this flag is set, the task will be
	   re-submitted to StarPU once it has been executed. This flag
	   must not be set if the flag starpu_task::destroy is set.
	   This flag must be set before making another task depend on
	   this one.

	   With starpu_task_insert() and alike this is set to 0.
	*/
	unsigned regenerate:1;

	/**
	   @private
	   This is only used for tasks that use multiformat handle.
	   This should only be used by StarPU.
	*/
	unsigned mf_skip:1;

	/**
	   do not allocate a submitorder id for this task

	   With starpu_task_insert() and alike this can be specified
	   thanks to ::STARPU_TASK_NO_SUBMITORDER followed by
	   an unsigned.
	*/
	unsigned no_submitorder:1;

	/**
	   Whether the scheduler has pushed the task on some queue

	   Set by StarPU.
	*/
	unsigned scheduled:1;
	unsigned prefetched:1;

	/**
	   Optional field. If the field
	   starpu_task::execute_on_a_specific_worker is set, this
	   field indicates the identifier of the worker that should
	   process this task (as returned by starpu_worker_get_id()).
	   This field is ignored if the field
	   starpu_task::execute_on_a_specific_worker is set to 0.

	   With starpu_task_insert() and alike this can be specified thanks to
	   ::STARPU_EXECUTE_ON_WORKER followed by an int.
	*/
	unsigned workerid;

	/**
	   Optional field. If the field
	   starpu_task::execute_on_a_specific_worker is set, this
	   field indicates the per-worker consecutive order in which
	   tasks should be executed on the worker. Tasks will be
	   executed in consecutive starpu_task::workerorder values,
	   thus ignoring the availability order or task priority. See
	   \ref StaticScheduling for more details. This field is
	   ignored if the field
	   starpu_task::execute_on_a_specific_worker is set to 0.

	   With starpu_task_insert() and alike this can be specified thanks to
	   ::STARPU_WORKER_ORDER followed by an unsigned.
	*/
	unsigned workerorder;

	/**
	   Optional field. If the field starpu_task::workerids_len is
	   different from 0, this field indicates an array of bits
	   (stored as uint32_t values) which indicate the set of
	   workers which are allowed to execute the task.
	   starpu_task::workerid takes precedence over this.

	   With starpu_task_insert() and alike, this can be specified
	   along the field workerids_len thanks to ::STARPU_TASK_WORKERIDS
	   followed by a number of workers and an array of bits which
	   size is the number of workers.
	*/
	uint32_t *workerids;

	/**
	   Optional field. This provides the number of uint32_t values
	   in the starpu_task::workerids array.

	   With starpu_task_insert() and alike, this can be specified
	   along the field workerids thanks to ::STARPU_TASK_WORKERIDS
	   followed by a number of workers and an array of bits which
	   size is the number of workers.
	*/
	unsigned workerids_len;

	/**
	   Optional field, the default value is ::STARPU_DEFAULT_PRIO.
	   This field indicates a level of priority for the task. This
	   is an integer value that must be set between the return
	   values of the function starpu_sched_get_min_priority() for
	   the least important tasks, and that of the function
	   starpu_sched_get_max_priority() for the most important
	   tasks (included). The ::STARPU_MIN_PRIO and
	   ::STARPU_MAX_PRIO macros are provided for convenience and
	   respectively return the value of
	   starpu_sched_get_min_priority() and
	   starpu_sched_get_max_priority(). Default priority is
	   ::STARPU_DEFAULT_PRIO, which is always defined as 0 in
	   order to allow static task initialization. Scheduling
	   strategies that take priorities into account can use this
	   parameter to take better scheduling decisions, but the
	   scheduling policy may also ignore it.

	   With starpu_task_insert() and alike this can be specified thanks to
	   ::STARPU_PRIORITY followed by an unsigned long long.
	*/
	int priority;

	/**
	   Current state of the task.

	   Set by StarPU.
	*/
	enum starpu_task_status status;

	/**
	   @private
	   This field is set when initializing a task. The function
	   starpu_task_submit() will fail if the field does not have
	   the correct value. This will hence avoid submitting tasks
	   which have not been properly initialised.
	*/
	int magic;

	/**
	   Allow to get the type of task, for filtering out tasks
	   in profiling outputs, whether it is really internal to
	   StarPU (::STARPU_TASK_TYPE_INTERNAL), a data acquisition
	   synchronization task (::STARPU_TASK_TYPE_DATA_ACQUIRE), or
	   a normal task (::STARPU_TASK_TYPE_NORMAL)

	   Set by StarPU.
	*/
	unsigned type;

	/**
	   color of the task to be used in dag.dot.

	   With starpu_task_insert() and alike this can be specified thanks to
	   ::STARPU_TASK_COLOR followed by an int.
	*/
	unsigned color;

	/**
	   Scheduling context.

	   With starpu_task_insert() and alike this can be specified thanks to
	   ::STARPU_SCHED_CTX followed by an unsigned.
	*/
	unsigned sched_ctx;

	/**
	   Help the hypervisor monitor the execution of this task.

	   With starpu_task_insert() and alike this can be specified thanks to
	   ::STARPU_HYPERVISOR_TAG followed by an int.
	*/
	int hypervisor_tag;

	/**
	   TODO: related with sched contexts and parallel tasks

	   With starpu_task_insert() and alike this can be specified thanks to
	   ::STARPU_POSSIBLY_PARALLEL followed by an unsigned.
	 */
	unsigned possibly_parallel;

	/**
	   Optional field. The bundle that includes this task. If no
	   bundle is used, this should be <c>NULL</c>.
	*/
	starpu_task_bundle_t bundle;

	/**
	   Optional field. Profiling information for the task.

	   With starpu_task_insert() and alike this can be specified thanks to
	   ::STARPU_TASK_PROFILING_INFO followed by a pointer to the
	   appropriate struct.
	*/
	struct starpu_profiling_task_info *profiling_info;

	/**
	   This can be set to the number of floating points operations
	   that the task will have to achieve. This is useful for
	   easily getting GFlops curves from the tool
	   <c>starpu_perfmodel_plot</c>, and for the hypervisor load
	   balancing.

	   With starpu_task_insert() and alike this can be specified thanks to
	   ::STARPU_FLOPS followed by a double.
	*/

	double flops;
	/**
	   Output field. Predicted duration of the task. This field is
	   only set if the scheduling strategy uses performance
	   models.

	   Set by StarPU.
	*/
	double predicted;

	/**
	   Output field. Predicted data transfer duration for the task in
	   microseconds. This field is only valid if the scheduling
	   strategy uses performance models.

	   Set by StarPU.
	*/
	double predicted_transfer;
	double predicted_start;

	/**
	   @private
	   A pointer to the previous task. This should only be used by
	   StarPU schedulers.
	*/
	struct starpu_task *prev;

	/**
	   @private
	   A pointer to the next task. This should only be used by
	   StarPU schedulers.
	*/
	struct starpu_task *next;

	/**
	   @private
	   This is private to StarPU, do not modify.
	*/
	void *starpu_private;

#ifdef STARPU_OPENMP
	/**
	   @private
	   This is private to StarPU, do not modify.
	 */
	struct starpu_omp_task *omp_task;
#else
	void *omp_task;
#endif
	/**
	   @private
	   This is private to StarPU, do not modify.
	  */
	unsigned nb_termination_call_required;

	/**
	   This field is managed by the scheduler, is it allowed to do
	   whatever with it.  Typically, some area would be allocated on push, and released on pop.

	   With starpu_task_insert() and alike this is set when using
	   ::STARPU_TASK_SCHED_DATA.
	*/
	void *sched_data;
};

/**
   To be used in the starpu_task::type field, for normal application tasks.
*/
#define STARPU_TASK_TYPE_NORMAL		0

/**
   To be used in the starpu_task::type field, for StarPU-internal tasks.
*/
#define STARPU_TASK_TYPE_INTERNAL	(1<<0)

/**
   To be used in the starpu_task::type field, for StarPU-internal data acquisition tasks.
*/
#define STARPU_TASK_TYPE_DATA_ACQUIRE	(1<<1)

/**
   Value to be used to initialize statically allocated tasks. This is
   equivalent to initializing a structure starpu_task
   with the function starpu_task_init().
*/
/* Note: remember to update starpu_task_init as well */
#define STARPU_TASK_INITIALIZER 			\
{							\
	.cl = NULL,					\
	.where = -1,					\
	.cl_arg = NULL,					\
	.cl_arg_size = 0,				\
	.callback_func = NULL,				\
	.callback_arg = NULL,				\
	.priority = STARPU_DEFAULT_PRIO,		\
	.use_tag = 0,					\
	.sequential_consistency = 1,			\
	.synchronous = 0,				\
	.execute_on_a_specific_worker = 0,		\
	.workerorder = 0,				\
	.bundle = NULL,					\
	.detach = 1,					\
	.destroy = 0,					\
	.regenerate = 0,				\
	.status = STARPU_TASK_INIT,			\
	.profiling_info = NULL,				\
	.predicted = NAN,				\
	.predicted_transfer = NAN,			\
	.predicted_start = NAN,				\
	.starpu_private = NULL,				\
	.magic = 42,                  			\
	.type = 0,					\
	.color = 0,					\
	.sched_ctx = STARPU_NMAX_SCHED_CTXS,		\
	.hypervisor_tag = 0,				\
	.flops = 0.0,					\
	.scheduled = 0,					\
	.prefetched = 0,				\
	.dyn_handles = NULL,				\
	.dyn_interfaces = NULL,				\
	.dyn_modes = NULL,				\
	.name = NULL,                        		\
	.possibly_parallel = 0                        	\
}

/**
   Return the number of buffers for \p task, i.e.
   starpu_codelet::nbuffers, or starpu_task::nbuffers if the former is
   \ref STARPU_VARIABLE_NBUFFERS.
*/
#define STARPU_TASK_GET_NBUFFERS(task) ((unsigned)((task)->cl->nbuffers == STARPU_VARIABLE_NBUFFERS ? ((task)->nbuffers) : ((task)->cl->nbuffers)))

/**
   Return the \p i -th data handle of \p task. If \p task is defined
   with a static or dynamic number of handles, will either return the
   \p i -th element of the field starpu_task::handles or the \p i -th
   element of the field starpu_task::dyn_handles (see \ref
   SettingManyDataHandlesForATask)
*/
#define STARPU_TASK_GET_HANDLE(task, i) (((task)->dyn_handles) ? (task)->dyn_handles[i] : (task)->handles[i])
#define STARPU_TASK_GET_HANDLES(task) (((task)->dyn_handles) ? (task)->dyn_handles : (task)->handles)

/**
   Set the \p i -th data handle of \p task with \p handle. If \p task
   is defined with a static or dynamic number of handles, will either
   set the \p i -th element of the field starpu_task::handles or the
   \p i -th element of the field starpu_task::dyn_handles
   (see \ref SettingManyDataHandlesForATask)
*/
#define STARPU_TASK_SET_HANDLE(task, handle, i)				\
	do { if ((task)->dyn_handles) (task)->dyn_handles[i] = handle; else (task)->handles[i] = handle; } while(0)

/**
   Return the access mode of the \p i -th data handle of \p codelet.
   If \p codelet is defined with a static or dynamic number of
   handles, will either return the \p i -th element of the field
   starpu_codelet::modes or the \p i -th element of the field
   starpu_codelet::dyn_modes (see \ref SettingManyDataHandlesForATask)
*/
#define STARPU_CODELET_GET_MODE(codelet, i) \
	(((codelet)->dyn_modes) ? (codelet)->dyn_modes[i] : (assert(i < STARPU_NMAXBUFS), (codelet)->modes[i]))

/**
   Set the access mode of the \p i -th data handle of \p codelet. If
   \p codelet is defined with a static or dynamic number of handles,
   will either set the \p i -th element of the field
   starpu_codelet::modes or the \p i -th element of the field
   starpu_codelet::dyn_modes (see \ref SettingManyDataHandlesForATask)
*/
#define STARPU_CODELET_SET_MODE(codelet, mode, i) \
	do { if ((codelet)->dyn_modes) (codelet)->dyn_modes[i] = mode; else (codelet)->modes[i] = mode; } while(0)

/**
   Return the access mode of the \p i -th data handle of \p task. If
   \p task is defined with a static or dynamic number of handles, will
   either return the \p i -th element of the field starpu_task::modes
   or the \p i -th element of the field starpu_task::dyn_modes (see
   \ref SettingManyDataHandlesForATask)
*/
#define STARPU_TASK_GET_MODE(task, i) \
	((task)->cl->nbuffers == STARPU_VARIABLE_NBUFFERS || (task)->dyn_modes ? \
	 (((task)->dyn_modes) ? (task)->dyn_modes[i] : (task)->modes[i]) : \
	 STARPU_CODELET_GET_MODE((task)->cl, i) )

/**
   Set the access mode of the \p i -th data handle of \p task. If \p
   task is defined with a static or dynamic number of handles, will
   either set the \p i -th element of the field starpu_task::modes or
   the \p i -th element of the field starpu_task::dyn_modes (see \ref
   SettingManyDataHandlesForATask)
*/
#define STARPU_TASK_SET_MODE(task, mode, i) \
	do {								\
		if ((task)->cl->nbuffers == STARPU_VARIABLE_NBUFFERS || (task)->cl->nbuffers > STARPU_NMAXBUFS) \
			if ((task)->dyn_modes) (task)->dyn_modes[i] = mode; else (task)->modes[i] = mode; \
		else							\
			STARPU_CODELET_SET_MODE((task)->cl, mode, i);	\
	} while(0)

/**
   Return the target node of the \p i -th data handle of \p codelet.
   If \p node is defined with a static or dynamic number of handles,
   will either return the \p i -th element of the field
   starpu_codelet::nodes or the \p i -th element of the field
   starpu_codelet::dyn_nodes (see \ref SettingManyDataHandlesForATask)
*/
#define STARPU_CODELET_GET_NODE(codelet, i) (((codelet)->dyn_nodes) ? (codelet)->dyn_nodes[i] : (codelet)->nodes[i])

/**
   Set the target node of the \p i -th data handle of \p codelet. If
   \p codelet is defined with a static or dynamic number of handles,
   will either set the \p i -th element of the field
   starpu_codelet::nodes or the \p i -th element of the field
   starpu_codelet::dyn_nodes (see \ref SettingManyDataHandlesForATask)
 */
#define STARPU_CODELET_SET_NODE(codelet, __node, i) \
	do { if ((codelet)->dyn_nodes) (codelet)->dyn_nodes[i] = __node; else (codelet)->nodes[i] = __node; } while(0)

/**
   Initialize \p task with default values. This function is implicitly
   called by starpu_task_create(). By default, tasks initialized with
   starpu_task_init() must be deinitialized explicitly with
   starpu_task_clean(). Tasks can also be initialized statically,
   using ::STARPU_TASK_INITIALIZER.
*/
void starpu_task_init(struct starpu_task *task);

/**
   Release all the structures automatically allocated to execute \p
   task, but not the task structure itself and values set by the user
   remain unchanged. It is thus useful for statically allocated tasks
   for instance. It is also useful when users want to execute the same
   operation several times with as least overhead as possible. It is
   called automatically by starpu_task_destroy(). It has to be called
   only after explicitly waiting for the task or after
   starpu_shutdown() (waiting for the callback is not enough, since
   StarPU still manipulates the task after calling the callback).
*/
void starpu_task_clean(struct starpu_task *task);

/**
   Allocate a task structure and initialize it with default values.
   Tasks allocated dynamically with starpu_task_create() are
   automatically freed when the task is terminated. This means that
   the task pointer can not be used any more once the task is
   submitted, since it can be executed at any time (unless
   dependencies make it wait) and thus freed at any time. If the field
   starpu_task::destroy is explicitly unset, the resources used by the
   task have to be freed by calling starpu_task_destroy().
*/
struct starpu_task *starpu_task_create(void) STARPU_ATTRIBUTE_MALLOC;

/**
   Free the resource allocated during starpu_task_create() and
   associated with \p task. This function is called automatically
   after the execution of a task when the field starpu_task::destroy
   is set, which is the default for tasks created by
   starpu_task_create(). Calling this function on a statically
   allocated task results in an undefined behaviour.
*/
void starpu_task_destroy(struct starpu_task *task);

/**
   Submit \p task to StarPU. Calling this function does not mean that
   the task will be executed immediately as there can be data or task
   (tag) dependencies that are not fulfilled yet: StarPU will take
   care of scheduling this task with respect to such dependencies.
   This function returns immediately if the field
   starpu_task::synchronous is set to 0, and block until the
   termination of the task otherwise. It is also possible to
   synchronize the application with asynchronous tasks by the means of
   tags, using the function starpu_tag_wait() function for instance.
   In case of success, this function returns 0, a return value of
   <c>-ENODEV</c> means that there is no worker able to process this
   task (e.g. there is no GPU available and this task is only
   implemented for CUDA devices). starpu_task_submit() can be called
   from anywhere, including codelet functions and callbacks, provided
   that the field starpu_task::synchronous is set to 0.
*/
int starpu_task_submit(struct starpu_task *task) STARPU_WARN_UNUSED_RESULT;

/**
   Submit \p task to the context \p sched_ctx_id. By default,
   starpu_task_submit() submits the task to a global context that is
   created automatically by StarPU.
*/
int starpu_task_submit_to_ctx(struct starpu_task *task, unsigned sched_ctx_id);

int starpu_task_finished(struct starpu_task *task) STARPU_WARN_UNUSED_RESULT;

/**
   Block until \p task has been executed. It is not possible to
   synchronize with a task more than once. It is not possible to wait
   for synchronous or detached tasks. Upon successful completion, this
   function returns 0. Otherwise, <c>-EINVAL</c> indicates that the
   specified task was either synchronous or detached.
*/
int starpu_task_wait(struct starpu_task *task) STARPU_WARN_UNUSED_RESULT;

/**
   Allow to wait for an array of tasks. Upon successful completion,
   this function returns 0. Otherwise, <c>-EINVAL</c> indicates that
   one of the tasks was either synchronous or detached.
*/
int starpu_task_wait_array(struct starpu_task **tasks, unsigned nb_tasks) STARPU_WARN_UNUSED_RESULT;

/**
   Block until all the tasks that were submitted (to the current
   context or the global one if there is no current context) are
   terminated. It does not destroy these tasks.
*/
int starpu_task_wait_for_all(void);

/**
   Block until there are \p n submitted tasks left (to the current
   context or the global one if there is no current context) to be
   executed. It does not destroy these tasks.
*/
int starpu_task_wait_for_n_submitted(unsigned n);

/**
   Wait until all the tasks that were already submitted to the context
   \p sched_ctx_id have been terminated.
*/
int starpu_task_wait_for_all_in_ctx(unsigned sched_ctx_id);

/**
   Wait until there are \p n tasks submitted left to be
   executed that were already submitted to the context \p
   sched_ctx_id.
*/
int starpu_task_wait_for_n_submitted_in_ctx(unsigned sched_ctx_id, unsigned n);

/**
   Wait until there is no more ready task.
*/
int starpu_task_wait_for_no_ready(void);

/**
   Return the number of submitted tasks which are ready for execution
   are already executing. It thus does not include tasks waiting for
   dependencies.
*/
int starpu_task_nready(void);

/**
   Return the number of submitted tasks which have not completed yet.
*/
int starpu_task_nsubmitted(void);

/**
   Set the iteration number for all the tasks to be submitted after
   this call. This is typically called at the beginning of a task
   submission loop. This number will then show up in tracing tools. A
   corresponding starpu_iteration_pop() call must be made to match the
   call to starpu_iteration_push(), at the end of the same task
   submission loop, typically.

   Nested calls to starpu_iteration_push() and starpu_iteration_pop()
   are allowed, to describe a loop nest for instance, provided that
   they match properly.
*/
void starpu_iteration_push(unsigned long iteration);

/**
   Drop the iteration number for submitted tasks. This must match a
   previous call to starpu_iteration_push(), and is typically called
   at the end of a task submission loop.
*/
void starpu_iteration_pop(void);

void starpu_do_schedule(void);

/**
   Initialize \p cl with default values. Codelets should preferably be
   initialized statically as shown in \ref DefiningACodelet. However
   such a initialisation is not always possible, e.g. when using C++.
*/
void starpu_codelet_init(struct starpu_codelet *cl);

/**
   Output on \c stderr some statistics on the codelet \p cl.
*/
void starpu_codelet_display_stats(struct starpu_codelet *cl);

/**
   Return the task currently executed by the worker, or <c>NULL</c> if
   it is called either from a thread that is not a task or simply
   because there is no task being executed at the moment.
*/
struct starpu_task *starpu_task_get_current(void);

/**
   Return the memory node number of parameter \p i of the task
   currently executed, or -1 if it is called either from a thread that
   is not a task or simply because there is no task being executed at
   the moment.

   Usually, the returned memory node number is simply the memory node
   for the current worker. That may however be different when using
   e.g. starpu_codelet::specific_nodes.
*/
int starpu_task_get_current_data_node(unsigned i);

/**
   Return the name of the performance model of \p task.
*/
const char *starpu_task_get_model_name(struct starpu_task *task);

/**
   Return the name of \p task, i.e. either its starpu_task::name
   field, or the name of the corresponding performance model.
*/
const char *starpu_task_get_name(struct starpu_task *task);

/**
   Allocate a task structure which is the exact duplicate of \p task.
*/
struct starpu_task *starpu_task_dup(struct starpu_task *task);

/**
   This function should be called by schedulers to specify the
   codelet implementation to be executed when executing \p task.
*/
void starpu_task_set_implementation(struct starpu_task *task, unsigned impl);

/**
   Return the codelet implementation to be executed
   when executing \p task.
*/
unsigned starpu_task_get_implementation(struct starpu_task *task);

/**
   Create and submit an empty task that unlocks a tag once all its
   dependencies are fulfilled.
 */
void starpu_create_sync_task(starpu_tag_t sync_tag, unsigned ndeps, starpu_tag_t *deps, void (*callback)(void *), void *callback_arg);

/**
   Create and submit an empty task with the given callback
 */
void starpu_create_callback_task(void (*callback)(void *), void *callback_arg);

/**
   Set the function to call when the watchdog detects that StarPU has
   not finished any task for STARPU_WATCHDOG_TIMEOUT seconds
*/
void starpu_task_watchdog_set_hook(void (*hook)(void *), void *hook_arg);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_TASK_H__ */
