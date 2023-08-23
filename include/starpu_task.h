/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2023 Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifdef STARPU_USE_CUDA
#include <cuda.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
   @defgroup API_Codelet_And_Tasks Codelet And Tasks
   @brief API to manipulate codelets and tasks.
   @{
*/

/**
   To be used when setting the field starpu_codelet::where to specify
   that the codelet has no computation part, and thus does not need to
   be scheduled, and data does not need to be actually loaded. This is
   thus essentially used for synchronization tasks.
*/
#define STARPU_NOWHERE ((1ULL) << 0)

/**
  Convert from enum starpu_worker_archtype to worker type mask for use in "where" fields
*/
#define STARPU_WORKER_TO_MASK(worker_archtype) (1ULL << (worker_archtype + 1))

/**
   To be used when setting the field starpu_codelet::where (or
   starpu_task::where) to specify the codelet (or the task) may be
   executed on a CPU processing unit.
*/
#define STARPU_CPU STARPU_WORKER_TO_MASK(STARPU_CPU_WORKER)

/**
   To be used when setting the field starpu_codelet::where (or
   starpu_task::where) to specify the codelet (or the task) may be
   executed on a CUDA processing unit.
*/
#define STARPU_CUDA STARPU_WORKER_TO_MASK(STARPU_CUDA_WORKER)

/**
   To be used when setting the field starpu_codelet::where (or
   starpu_task::where) to specify the codelet (or the task) may be
   executed on a HIP processing unit.
*/
#define STARPU_HIP STARPU_WORKER_TO_MASK(STARPU_HIP_WORKER)

/**
   To be used when setting the field starpu_codelet::where (or
   starpu_task::where) to specify the codelet (or the task) may be
   executed on a OpenCL processing unit.
*/
#define STARPU_OPENCL STARPU_WORKER_TO_MASK(STARPU_OPENCL_WORKER)

/**
   To be used when setting the field starpu_codelet::where (or
   starpu_task::where) to specify the codelet (or the task) may be
   executed on a MAX FPGA.
*/
#define STARPU_MAX_FPGA STARPU_WORKER_TO_MASK(STARPU_MAX_FPGA_WORKER)

/**
   To be used when setting the field starpu_codelet::where (or
   starpu_task::where) to specify the codelet (or the task) may be
   executed on a MPI Slave processing unit.
*/
#define STARPU_MPI_MS STARPU_WORKER_TO_MASK(STARPU_MPI_MS_WORKER)

/**
   To be used when setting the field starpu_codelet::where (or
   starpu_task::where) to specify the codelet (or the task) may be
   executed on a TCP/IP Slave processing unit.
*/
#define STARPU_TCPIP_MS STARPU_WORKER_TO_MASK(STARPU_TCPIP_MS_WORKER)

/**
   Value to be set in starpu_codelet::flags to execute the codelet
   functions even in simgrid mode.
*/
#define STARPU_CODELET_SIMGRID_EXECUTE (1 << 0)

/**
   Value to be set in starpu_codelet::flags to execute the codelet
   functions even in simgrid mode, and later inject the measured
   timing inside the simulation.
*/
#define STARPU_CODELET_SIMGRID_EXECUTE_AND_INJECT (1 << 1)

/**
   Value to be set in starpu_codelet::flags to make starpu_task_submit()
   not submit automatic asynchronous partitioning/unpartitioning.
*/
#define STARPU_CODELET_NOPLANS (1 << 2)

/**
   Value to be set in starpu_codelet::cuda_flags to allow asynchronous
   CUDA kernel execution.
*/
#define STARPU_CUDA_ASYNC (1 << 0)

/**
   Value to be set in starpu_codelet::hip_flags to allow asynchronous
   HIP kernel execution.
*/
#define STARPU_HIP_ASYNC (1 << 0)

/**
   Value to be set in starpu_codelet::opencl_flags to allow
   asynchronous OpenCL kernel execution.
*/
#define STARPU_OPENCL_ASYNC (1 << 0)

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
	STARPU_SEQ = 0, /**< (default) for classical sequential
			    tasks.
			 */
	STARPU_SPMD,	/**< for a parallel task whose threads are
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

/**
   todo
*/
enum starpu_task_status
{
	STARPU_TASK_INIT, /**< The task has just been initialized. */
#define STARPU_TASK_INIT    0
#define STARPU_TASK_INVALID STARPU_TASK_INIT /**< old name for STARPU_TASK_INIT */
	STARPU_TASK_BLOCKED,		     /**< The task has just been submitted, and its dependencies has not been checked yet. */
	STARPU_TASK_READY,		     /**< The task is ready for execution. */
	STARPU_TASK_RUNNING,		     /**< The task is running on some worker. */
	STARPU_TASK_FINISHED,		     /**< The task is finished executing. */
	STARPU_TASK_BLOCKED_ON_TAG,	     /**< The task is waiting for a tag. */
	STARPU_TASK_BLOCKED_ON_TASK,	     /**< The task is waiting for a task. */
	STARPU_TASK_BLOCKED_ON_DATA,	     /**< The task is waiting for some data. */
	STARPU_TASK_STOPPED		     /**< The task is stopped. */
};

/**
   CPU implementation of a codelet.
*/
typedef void (*starpu_cpu_func_t)(void **, void *);

/**
   CUDA implementation of a codelet.
*/
typedef void (*starpu_cuda_func_t)(void **, void *);

/**
   HIP implementation of a codelet.
*/
typedef void (*starpu_hip_func_t)(void **, void *);

/**
   OpenCL implementation of a codelet.
*/
typedef void (*starpu_opencl_func_t)(void **, void *);

/**
   Maxeler FPGA implementation of a codelet.
*/
typedef void (*starpu_max_fpga_func_t)(void **, void *);

/**
   @ingroup API_Bubble Hierarchical Dags
   Bubble decision function
*/
typedef int (*starpu_bubble_func_t)(struct starpu_task *t, void *arg);

/**
   @ingroup API_Bubble Hierarchical Dags
   Bubble DAG generation function
*/
typedef void (*starpu_bubble_gen_dag_func_t)(struct starpu_task *t, void *arg);

/**
   @deprecated
   Setting the field starpu_codelet::cpu_func with this macro
   indicates the codelet will have several implementations. The use of
   this macro is deprecated. One should always only define the field
   starpu_codelet::cpu_funcs.
*/
#define STARPU_MULTIPLE_CPU_IMPLEMENTATIONS ((starpu_cpu_func_t)-1)

/**
   @deprecated
   Setting the field starpu_codelet::cuda_func with this macro
   indicates the codelet will have several implementations. The use of
   this macro is deprecated. One should always only define the field
   starpu_codelet::cuda_funcs.
*/
#define STARPU_MULTIPLE_CUDA_IMPLEMENTATIONS ((starpu_cuda_func_t)-1)

/**
   @deprecated
   Setting the field starpu_codelet::hip_func with this macro
   indicates the codelet will have several implementations. The use of
   this macro is deprecated. One should always only define the field
   starpu_codelet::hip_funcs.
*/
#define STARPU_MULTIPLE_HIP_IMPLEMENTATIONS ((starpu_hip_func_t)-1)

/**
   @deprecated
   Setting the field starpu_codelet::opencl_func with this macro
   indicates the codelet will have several implementations. The use of
   this macro is deprecated. One should always only define the field
   starpu_codelet::opencl_funcs.
*/
#define STARPU_MULTIPLE_OPENCL_IMPLEMENTATIONS ((starpu_opencl_func_t)-1)

/**
   Value to set in starpu_codelet::nbuffers to specify that the
   codelet can accept a variable number of buffers, specified in
   starpu_task::nbuffers.
*/
#define STARPU_VARIABLE_NBUFFERS (-1)

/**
   Value to be set in the starpu_codelet::nodes field to request
   StarPU to put the data in local memory of the worker running the task (this
   is the default behavior).
*/
#define STARPU_SPECIFIC_NODE_LOCAL (-1)

/**
   Value to be set in the starpu_codelet::nodes field to request
   StarPU to put the data in CPU-accessible memory (and let StarPU
   choose the NUMA node).
*/
#define STARPU_SPECIFIC_NODE_CPU (-2)

/**
   Value to be set in the starpu_codelet::nodes field to request
   StarPU to put the data in some slow memory.
*/
#define STARPU_SPECIFIC_NODE_SLOW (-3)

/**
   Value to be set in the starpu_codelet::nodes field to request
   StarPU to put the data in some fast memory.
*/
#define STARPU_SPECIFIC_NODE_FAST (-4)

/**
   Value to be set in the starpu_codelet::nodes field to let StarPU decide
   whether to put the data in the local memory of the worker running the task,
   or in CPU-accessible memory (and let StarPU choose the NUMA node).
*/
#define STARPU_SPECIFIC_NODE_LOCAL_OR_CPU (-5)

/**
   Value to be set in the starpu_codelet::nodes field to make StarPU not actually
   put the data in any particular memory, i.e. the task will only get the
   sequential consistency dependencies, but not actually trigger any data
   transfer.
*/
#define STARPU_SPECIFIC_NODE_NONE (-6)

struct starpu_transaction;
struct _starpu_trs_epoch;
typedef struct _starpu_trs_epoch *starpu_trs_epoch_t;
struct starpu_task;

/**
   The codelet structure describes a kernel that is possibly
   implemented on various targets. For compatibility, make sure to
   initialize the whole structure to zero, either by using explicit
   memset, or the function starpu_codelet_init(), or by letting the
   compiler implicitly do it in e.g. static storage case.

   Note that the codelet structure needs to exist until the task is
   terminated. If dynamic codelet allocation is desired, release should be done
   no sooner than the starpu_task::callback_func callback time.

   If the application wants to make the structure constant, it needs to be
   filled exactly as StarPU expects:

   - starpu_codelet::cpu_funcs, starpu_codelet::cuda_funcs, etc. must be used instead
   of the deprecated starpu_codelet::cpu_func, starpu_codelet::cuda_func, etc.

   - the starpu_codelet::where field must be set.

   and additionally, starpu_codelet::checked must be set to 1 to tell StarPU
   that the conditions above are properly met. Also, the \ref
   STARPU_CODELET_PROFILING environment variable must be set to 0.
   An example is provided in tests/main/const_codelet.c
*/
struct starpu_codelet
{
	/**
	   Optional field to indicate which types of processing units
	   are able to execute the codelet. The different values
	   ::STARPU_CPU, ::STARPU_CUDA, ::STARPU_HIP, ::STARPU_OPENCL can be
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
	   Optional array of function pointers to the HIP
	   implementations of the codelet. The functions must be
	   host-functions written in the HIP runtime API. Their
	   prototype must be:
	   \code{.c}
	   void hip_func(void *buffers[], void *cl_arg)
	   \endcode
	   If the field starpu_codelet::where is set, then the field
	   starpu_codelet::hip_funcs is ignored if ::STARPU_HIP does
	   not appear in the field starpu_codelet::where, it must be
	   non-<c>NULL</c> otherwise.
	*/
	starpu_hip_func_t hip_funcs[STARPU_MAXIMPLEMENTATIONS];

	/**
	   Optional array of flags for HIP execution. They specify
	   some semantic details about HIP kernel execution, such as
	   asynchronous execution.
	*/
	char hip_flags[STARPU_MAXIMPLEMENTATIONS];

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
           Optional array of function pointers to the Maxeler FPGA
           implementations of the codelet. The functions prototype
           must be:
           \code{.c}
           void fpga_func(void *buffers[], void *cl_arg)
           \endcode
           The first argument being the array of data managed by the
           data management library, and the second argument is a
           pointer to the argument passed from the field
           starpu_task::cl_arg. If the field starpu_codelet::where is
           set, then the field starpu_codelet::max_fpga_funcs is ignored if
           ::STARPU_MAX_FPGA does not appear in the field
           starpu_codelet::where, it must be non-<c>NULL</c> otherwise.
        */
	starpu_max_fpga_func_t max_fpga_funcs[STARPU_MAXIMPLEMENTATIONS];

	/**
	   Optional array of strings which provide the name of the CPU
	   functions referenced in the array
	   starpu_codelet::cpu_funcs. This can be used when running on
	   MPI MS devices for StarPU to simply look
	   up the MPI MS function implementation through its name.
	*/
	const char *cpu_funcs_name[STARPU_MAXIMPLEMENTATIONS];

	/**
	   Optional function to decide if the task is to be
	   transformed into a bubble
	 */
	starpu_bubble_func_t bubble_func;

	/**
	   Optional function to transform the task into a new graph
	 */
	starpu_bubble_gen_dag_func_t bubble_gen_dag_func;

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
	   required access modes to the data needed by the codelet
	   (e.g. ::STARPU_RW). The number of entries in this array
	   must be specified in the field starpu_codelet::nbuffers,
	   and should not exceed \ref STARPU_NMAXBUFS. If
	   insufficient, this value can be set with the configure
	   option \ref enable-maxbuffers "--enable-maxbuffers".
	*/
	enum starpu_data_access_mode modes[STARPU_NMAXBUFS];

	/**
	   Is an array of ::starpu_data_access_mode. It describes the
	   required access modes to the data needed by the codelet
	   (e.g. ::STARPU_RW). The number of entries in this array
	   must be specified in the field starpu_codelet::nbuffers.
	   This field should be used for codelets having a number of
	   data greater than \ref STARPU_NMAXBUFS (see \ref
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
	   used for codelets having a number of data greater than
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
	   model associated to this codelet (in J). This optional field is
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
	   debugging purposes. Value 0 acts like if this field wasn't specified.
	   Color representation is hex triplet (for example: 0xff0000 is red,
	   0x0000ff is blue, 0xffa500 is orange, ...).
	*/
	unsigned color;

	/**
	   Optional field, the default value is <c>NULL</c>. This is a
	   function pointer of prototype <c>void (*f)(void *)</c>
	   which specifies a possible callback. If this pointer is
	   non-<c>NULL</c>, the callback function is executed on the
	   host after the execution of the task. If the task defines a
	   callback, the codelet callback is not called, unless called
	   within the task callback function.
	   The callback is passed the value contained in the
	   starpu_task::callback_arg field. No callback is executed if
	   the field is set to <c>NULL</c>.
	*/
	void (*callback_func)(void *);

	/**
	   Various flags for the codelet.
	 */
	int flags;

	struct starpu_perf_counter_sample *perf_counter_sample;
	struct starpu_perf_counter_sample_cl_values *perf_counter_values;

	/**
	   Whether _starpu_codelet_check_deprecated_fields was already done or not.
	 */
	int checked;
};

/**
   Codelet with empty function defined for all drivers
*/
extern struct starpu_codelet starpu_codelet_nop;

/**
   Describe a data handle along with an access mode.
*/
struct starpu_data_descr
{
	starpu_data_handle_t handle;	   /**< data */
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
	   Store the iteration numbers (as defined by
	   starpu_iteration_push() / starpu_iteration_pop()) during
	   task submission.
	*/
	long iterations[2];

	/**
	   Optional file name where the task was submitted. This can be useful
	   for debugging purposes.
	*/
	const char *file;

	/**
	  Optional line number where the task was submitted. This can be useful
	   for debugging purposes.
	*/
	int line;

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

	/**
	   Keep dyn_handles, dyn_interfaces and dyn_modes before the
	   equivalent static arrays, so we can detect dyn_handles
	   being NULL while nbuffers being bigger that STARPU_NMAXBUFS
	   (otherwise the overflow would put a non-NULL)
	*/

	/**
	   Array of ::starpu_data_handle_t. Specify the handles to the
	   different pieces of data accessed by the task. The number
	   of entries in this array must be specified in the field
	   starpu_codelet::nbuffers. This field should be used for
	   tasks having a number of data greater than \ref
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
	   data greater than \ref STARPU_NMAXBUFS (see \ref
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
	   \ref STARPU_NMAXBUFS. If insufficient, this value can be
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
	   required access modes to the data needed by the codelet
	   (e.g. ::STARPU_RW). The number of entries in this array
	   must be specified in the field starpu_task::nbuffers, and
	   should not exceed \ref STARPU_NMAXBUFS. If insufficient,
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
	   Optional pointer which points to the return value of submitted task.
	   The default value is <c>NULL</c>. starpu_codelet_pack_arg()
	   and starpu_codelet_unpack_arg() can be used to respectively
	   pack and unpack the return value into and form it. starpu_task::cl_ret
	   can be used for MPI support. The only requirement is that
	   the size of the return value must be set in starpu_task::cl_ret_size .
	*/
	void *cl_ret;

	/**
	   Optional field. The buffer of starpu_codelet_pack_arg()
	   and starpu_codelet_unpack_arg() can be allocated with
	   the starpu_task::cl_ret_size bytes starting at address starpu_task::cl_ret.
	   starpu_task::cl_ret_size can be used for MPI support.
	*/
	size_t cl_ret_size;

	/**
	   Optional field, the default value is <c>NULL</c>. This is a
	   function pointer of prototype <c>void (*f)(void *)</c> which
	   specifies a possible callback. If this pointer is non-<c>NULL</c>,
	   the callback function is executed on the host after the execution of
	   the task. Contrary to starpu_task::callback_func, it is called
	   before releasing tasks which depend on this task, so those cannot be
	   already executing. The callback is passed
	   the value contained in the starpu_task::epilogue_callback_arg field.
	   No callback is executed if the field is set to <c>NULL</c>.

	   With starpu_task_insert() and alike this can be specified thanks to
	   ::STARPU_EPILOGUE_CALLBACK followed by the function pointer.
	*/
	void (*epilogue_callback_func)(void *);

	/**
	   Optional field, the default value is <c>NULL</c>. This is
	   the pointer passed to the epilogue callback function. This field is
	   ignored if the field starpu_task::epilogue_callback_func is set to
	   <c>NULL</c>.
	*/
	void *epilogue_callback_arg;

	/**
	   Optional field, the default value is <c>NULL</c>. This is a
	   function pointer of prototype <c>void (*f)(void *)</c>
	   which specifies a possible callback. If this pointer is
	   non-<c>NULL</c>, the callback function is executed on the
	   host after the execution of the task. Contrary to
	   starpu_task::epilogue_callback, it is called after releasing
	   tasks which depend on this task, so those
	   might already be executing. The callback is passed the
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
	   ::STARPU_CALLBACK_ARG followed by the argument pointer, or thanks to
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

	/**
	   Optional field, the default value is <c>NULL</c>. This is a
	   function pointer of prototype <c>void (*f)(void*)</c>
	   which specifies a possible callback. If this pointer is
	   non-<c>NULL</c>, the callback function is executed on the host
	   when the task is pop-ed from the scheduler, just before getting
	   executed. The callback is passed the value contained in the
	   starpu_task::prologue_callback_pop_arg field.
	   No callback is executed if the field is set to <c>NULL</c>.

	   With starpu_task_insert() and alike this can be specified thanks to
	   ::STARPU_PROLOGUE_CALLBACK_POP followed by the function pointer.
	*/
	void (*prologue_callback_pop_func)(void *);

	/**
	   Optional field, the default value is <c>NULL</c>. This is
	   the pointer passed to the prologue_callback_pop function. This
	   field is ignored if the field
	   starpu_task::prologue_callback_pop_func is set to <c>NULL</c>.

	   With starpu_task_insert() and alike this can be specified thanks to
	   ::STARPU_PROLOGUE_CALLBACK_POP_ARG followed by the argument.
	   */
	void *prologue_callback_pop_arg;

	/**
	   Transaction to which the task belongs, if any
	*/
	struct starpu_transaction *transaction;

	/**
	   Transaction epoch to which the task belongs, if any
	*/
	starpu_trs_epoch_t trs_epoch;

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
	   saves the user from defining a callback just for that.

	   With starpu_task_insert() and alike this is set to 1 when using
	   ::STARPU_CL_ARGS.
	*/
	unsigned cl_arg_free : 1;

	/**
	   Optional field. In case starpu_task::cl_ret was allocated
	   by the application through <c>malloc()</c>, setting
	   starpu_task::cl_ret_free to 1 makes StarPU automatically
	   call <c>free(cl_ret)</c> when destroying the task.
	*/
	unsigned cl_ret_free : 1;

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
	unsigned callback_arg_free : 1;

	/**
	   Optional field. In case starpu_task::epilogue_callback_arg was
	   allocated by the application through <c>malloc()</c>,
	   setting starpu_task::epilogue_callback_arg_free to 1 makes StarPU
	   automatically call <c>free(epilogue_callback_arg)</c> when
	   destroying the task.
	*/
	unsigned epilogue_callback_arg_free : 1;

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
	unsigned prologue_callback_arg_free : 1;

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
	unsigned prologue_callback_pop_arg_free : 1;

	/**
	   Optional field, the default value is 0. If set, this flag
	   indicates that the task should be associated with the tag
	   contained in the starpu_task::tag_id field. Tag allow the
	   application to synchronize with the task and to express
	   task dependencies easily.

	   With starpu_task_insert() and alike this is set to 1 when using
	   ::STARPU_TAG.
	*/
	unsigned use_tag : 1;

	/**
	   If this flag is set (which is the default), sequential
	   consistency is enforced for the data parameters of this
	   task for which sequential consistency is enabled. Clearing
	   this flag permits to disable sequential consistency for
	   this task, even if data have it enabled.

	   With starpu_task_insert() and alike this can be specified thanks to
	   ::STARPU_SEQUENTIAL_CONSISTENCY followed by an unsigned.
	*/
	unsigned sequential_consistency : 1;

	/**
	   If this flag is set, the function starpu_task_submit() is
	   blocking and returns only when the task has been executed
	   (or if no worker is able to process the task). Otherwise,
	   starpu_task_submit() returns immediately.

	   With starpu_task_insert() and alike this can be specified thanks to
	   ::STARPU_TASK_SYNCHRONOUS followed an int.
	*/
	unsigned synchronous : 1;

	/**
	   Default value is 0. If this flag is set, StarPU will bypass
	   the scheduler and directly affect this task to the worker
	   specified by the field starpu_task::workerid.

	   With starpu_task_insert() and alike this is set to 1 when using
	   ::STARPU_EXECUTE_ON_WORKER.
	*/
	unsigned execute_on_a_specific_worker : 1;

	/**
	   Optional field, default value is 1. If this flag is set, it
	   is not possible to synchronize with the task by the means
	   of starpu_task_wait() later on. Internal data structures
	   are only guaranteed to be freed once starpu_task_wait() is
	   called if the flag is not set.

	   With starpu_task_insert() and alike this is set to 1.
	*/
	unsigned detach : 1;

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

	   Calling starpu_task_set_destroy() can be used to set this field to 1 after submission.
	   Indeed this function will manage concurrency against the termination of the task.
	*/
	unsigned destroy : 1;

	/**
	   Optional field. If this flag is set, the task will be
	   re-submitted to StarPU once it has been executed. This flag
	   must not be set if the flag starpu_task::destroy is set.
	   This flag must be set before making another task depend on
	   this one.

	   With starpu_task_insert() and alike this is set to 0.
	*/
	unsigned regenerate : 1;

	/**
	   do not allocate a submitorder id for this task

	   With starpu_task_insert() and alike this can be specified
	   thanks to ::STARPU_TASK_NO_SUBMITORDER followed by
	   an unsigned.
	*/
	unsigned no_submitorder : 1;

	/**
	   @private
	   This is only used for tasks that use multiformat handle.
	   This should only be used by StarPU.
	*/
	unsigned char mf_skip;

	/**
	   Whether this task has failed and will thus have to be retried

	   Set by StarPU.
	*/
	unsigned char failed;

	/**
	   Whether the scheduler has pushed the task on some queue

	   Set by StarPU.
	*/
	unsigned char scheduled;

	/**
	   Whether the scheduler has prefetched the task's data

	   Set by StarPU.
	*/
	unsigned char prefetched;

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

	   Call starpu_task_status_get_as_string() to get the status as a string.

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
	   The application can set this to the number of floating points
	   operations that the task will have to achieve. StarPU will measure
	   the time that the task takes, and divide the two to get the GFlop/s
	   achieved by the task.  This will allow getting GFlops/s curves
	   from the tool <c>starpu_perfmodel_plot</c>, and is useful for the
	   hypervisor load balancing.

	   With starpu_task_insert() and alike this can be specified thanks to
	   ::STARPU_FLOPS followed by a double.
	*/

	double flops;
	/**
	   Output field. Predicted duration of the task in microseconds. This field is
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
	   When using hierarchical dags, the job identifier of the
	   bubble task which created the current task
	*/
	unsigned long bubble_parent;

	/**
	   When using hierarchical dags, a pointer to the bubble
	   decision function
	*/
	starpu_bubble_func_t bubble_func;

	/**
	   When using hierarchical dags, a pointer to an argument to
	   be given when calling the bubble decision function
	*/
	void *bubble_func_arg;

	/**
	   When using hierarchical dags, a pointer to the bubble
	   DAG generation function
	*/
	starpu_bubble_gen_dag_func_t bubble_gen_dag_func;

	/**
	   When using hierarchical dags, a pointer to an argument to
	   be given when calling the bubble DAG generation function
	 */
	void *bubble_gen_dag_func_arg;

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
#define STARPU_TASK_TYPE_NORMAL 0

/**
   To be used in the starpu_task::type field, for StarPU-internal tasks.
*/
#define STARPU_TASK_TYPE_INTERNAL (1 << 0)

/**
   To be used in the starpu_task::type field, for StarPU-internal data acquisition tasks.
*/
#define STARPU_TASK_TYPE_DATA_ACQUIRE (1 << 1)

/* Note: remember to update starpu_task_init and starpu_task_ft_create_retry
 * as well */
/**
   Value to be used to initialize statically allocated tasks. This is
   equivalent to initializing a structure starpu_task
   with the function starpu_task_init().
*/
#define STARPU_TASK_INITIALIZER                                         \
	{                                                               \
		.cl			      = NULL,                   \
		.where			      = -1,                     \
		.cl_arg			      = NULL,                   \
		.cl_arg_size		      = 0,                      \
		.cl_ret			      = NULL,                   \
		.cl_ret_size		      = 0,                      \
		.callback_func		      = NULL,                   \
		.callback_arg		      = NULL,                   \
		.epilogue_callback_func	      = NULL,                   \
		.epilogue_callback_arg	      = NULL,                   \
		.priority		      = STARPU_DEFAULT_PRIO,    \
		.use_tag		      = 0,                      \
		.sequential_consistency	      = 1,                      \
		.synchronous		      = 0,                      \
		.execute_on_a_specific_worker = 0,                      \
		.workerorder		      = 0,                      \
		.bundle			      = NULL,                   \
		.detach			      = 1,                      \
		.destroy		      = 0,                      \
		.regenerate		      = 0,                      \
		.status			      = STARPU_TASK_INIT,       \
		.profiling_info		      = NULL,                   \
		.predicted		      = NAN,                    \
		.predicted_transfer	      = NAN,                    \
		.predicted_start	      = NAN,                    \
		.starpu_private		      = NULL,                   \
		.magic			      = 42,                     \
		.type			      = 0,                      \
		.color			      = 0,                      \
		.sched_ctx		      = STARPU_NMAX_SCHED_CTXS, \
		.hypervisor_tag		      = 0,                      \
		.flops			      = 0.0,                    \
		.scheduled		      = 0,                      \
		.prefetched		      = 0,                      \
		.dyn_handles		      = NULL,                   \
		.dyn_interfaces		      = NULL,                   \
		.dyn_modes		      = NULL,                   \
		.name			      = NULL,                   \
		.possibly_parallel	      = 0                       \
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

/**
   Return all the data handles of \p task. If \p task is defined
   with a static or dynamic number of handles, will either return all
   the element of the field starpu_task::handles or all the elements
   of the field starpu_task::dyn_handles (see \ref SettingManyDataHandlesForATask)
*/
#define STARPU_TASK_GET_HANDLES(task) (((task)->dyn_handles) ? (task)->dyn_handles : (task)->handles)

/**
   Set the \p i -th data handle of \p task with \p handle. If \p task
   is defined with a static or dynamic number of handles, will either
   set the \p i -th element of the field starpu_task::handles or the
   \p i -th element of the field starpu_task::dyn_handles
   (see \ref SettingManyDataHandlesForATask)
*/
#define STARPU_TASK_SET_HANDLE(task, handle, i)          \
	do {                                             \
		if ((task)->dyn_handles)                 \
			(task)->dyn_handles[i] = handle; \
		else                                     \
			(task)->handles[i] = handle;     \
	}                                                \
	while (0)

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
#define STARPU_CODELET_SET_MODE(codelet, mode, i)       \
	do {                                            \
		if ((codelet)->dyn_modes)               \
			(codelet)->dyn_modes[i] = mode; \
		else                                    \
			(codelet)->modes[i] = mode;     \
	}                                               \
	while (0)

/**
   Return the access mode of the \p i -th data handle of \p task. If
   \p task is defined with a static or dynamic number of handles, will
   either return the \p i -th element of the field starpu_task::modes
   or the \p i -th element of the field starpu_task::dyn_modes (see
   \ref SettingManyDataHandlesForATask)
*/
#define STARPU_TASK_GET_MODE(task, i) \
	((task)->cl->nbuffers == STARPU_VARIABLE_NBUFFERS || (task)->dyn_modes ? (((task)->dyn_modes) ? (task)->dyn_modes[i] : (task)->modes[i]) : STARPU_CODELET_GET_MODE((task)->cl, i))

/**
   Set the access mode of the \p i -th data handle of \p task. If \p
   task is defined with a static or dynamic number of handles, will
   either set the \p i -th element of the field starpu_task::modes or
   the \p i -th element of the field starpu_task::dyn_modes (see \ref
   SettingManyDataHandlesForATask)
*/
#define STARPU_TASK_SET_MODE(task, mode, i)                                                                                          \
	do {                                                                                                                         \
		if ((task)->cl->nbuffers == STARPU_VARIABLE_NBUFFERS || (task)->cl->nbuffers > STARPU_NMAXBUFS)                      \
			if ((task)->dyn_modes)                                                                                       \
				(task)->dyn_modes[i] = mode;                                                                         \
			else                                                                                                         \
				(task)->modes[i] = mode;                                                                             \
		else                                                                                                                 \
		{                                                                                                                    \
			enum starpu_data_access_mode cl_mode = STARPU_CODELET_GET_MODE((task)->cl, i);                               \
			STARPU_ASSERT_MSG(cl_mode == mode,                                                                           \
					  "Task <%s> can't set its  %d-th buffer mode to %d as the codelet it derives from uses %d", \
					  (task)->cl->name, i, mode, cl_mode);                                                       \
		}                                                                                                                    \
	}                                                                                                                            \
	while (0)

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
#define STARPU_CODELET_SET_NODE(codelet, __node, i)       \
	do {                                              \
		if ((codelet)->dyn_nodes)                 \
			(codelet)->dyn_nodes[i] = __node; \
		else                                      \
			(codelet)->nodes[i] = __node;     \
	}                                                 \
	while (0)

/**
   Initialize \p task with default values. This function is implicitly
   called by starpu_task_create(). By default, tasks initialized with
   starpu_task_init() must be deinitialized explicitly with
   starpu_task_clean(). Tasks can also be initialized statically,
   using ::STARPU_TASK_INITIALIZER.
   See \ref PerformanceModelCalibration for more details.
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
   See \ref PerformanceModelCalibration for more details.
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
   See \ref SubmittingATask for more details.
*/
struct starpu_task *starpu_task_create(void) STARPU_ATTRIBUTE_MALLOC;

/**
   Allocate a task structure that does nothing but accesses data \p handle
   with mode \p mode. This allows to synchronize with the task graph, according
   to the sequential consistency, against tasks submitted before or after
   submitting this task. One can then use starpu_task_declare_deps_array() or
   starpu_task_end_dep_add() / starpu_task_end_dep_release() to add dependencies
   against this task before submitting it.
   See \ref SynchronizationTasks for more details.
 */
struct starpu_task *starpu_task_create_sync(starpu_data_handle_t handle, enum starpu_data_access_mode mode) STARPU_ATTRIBUTE_MALLOC;

/**
   Free the resource allocated during starpu_task_create() and
   associated with \p task. This function is called automatically
   after the execution of a task when the field starpu_task::destroy
   is set, which is the default for tasks created by
   starpu_task_create(). Calling this function on a statically
   allocated task results in an undefined behaviour.
   See \ref Per-taskFeedback and \ref PerformanceModelExample for more details.
*/
void starpu_task_destroy(struct starpu_task *task);

/**
   Tell StarPU to free the resources associated with \p task when the task is
   over. This is equivalent to having set task->destroy = 1 before submission,
   the difference is that this can be called after submission and properly deals
   with concurrency with the task execution.
   See \ref WaitingForTasks for more details.
*/
void starpu_task_set_destroy(struct starpu_task *task);

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
   See \ref SubmittingATask for more details.
*/
int starpu_task_submit(struct starpu_task *task) STARPU_WARN_UNUSED_RESULT;

#ifdef STARPU_USE_FXT
static inline int starpu_task_submit_line(struct starpu_task *task, const char *file, int line)
{
	task->file = file;
	task->line = line;
	return starpu_task_submit(task);
}
#define starpu_task_submit(task) starpu_task_submit_line((task), __FILE__, __LINE__)
#endif

/**
   Submit \p task to StarPU with dependency bypass.

   This can only be called on behalf of another task which has already taken the
   proper dependencies, e.g. this task is just an attempt of doing the actual
   computation of that task.
   See \ref TaskRetry for more details.
*/
int starpu_task_submit_nodeps(struct starpu_task *task) STARPU_WARN_UNUSED_RESULT;

/**
   Submit \p task to the context \p sched_ctx_id. By default,
   starpu_task_submit() submits the task to a global context that is
   created automatically by StarPU.
   See \ref SubmittingTasksToAContext for more details.
*/
int starpu_task_submit_to_ctx(struct starpu_task *task, unsigned sched_ctx_id);

/**
   Return 1 if \p task is terminated.
   See \ref WaitingForTasks for more details.
*/
int starpu_task_finished(struct starpu_task *task) STARPU_WARN_UNUSED_RESULT;

/**
   Block until \p task has been executed. It is not possible to
   synchronize with a task more than once. It is not possible to wait
   for synchronous or detached tasks. Upon successful completion, this
   function returns 0. Otherwise, <c>-EINVAL</c> indicates that the
   specified task was either synchronous or detached.
   See \ref SubmittingATask for more details.
*/
int starpu_task_wait(struct starpu_task *task) STARPU_WARN_UNUSED_RESULT;

/**
   Allow to wait for an array of tasks. Upon successful completion,
   this function returns 0. Otherwise, <c>-EINVAL</c> indicates that
   one of the tasks was either synchronous or detached.
   See \ref WaitingForTasks for more details.
*/
int starpu_task_wait_array(struct starpu_task **tasks, unsigned nb_tasks) STARPU_WARN_UNUSED_RESULT;

/**
   Block until all the tasks that were submitted (to the current
   context or the global one if there is no current context) are
   terminated. It does not destroy these tasks.
   See \ref SubmittingATask for more details.
*/
int starpu_task_wait_for_all(void);

/**
   Block until there are \p n submitted tasks left (to the current
   context or the global one if there is no current context) to be
   executed. It does not destroy these tasks.
   See \ref HowtoReuseMemory for more details.
*/
int starpu_task_wait_for_n_submitted(unsigned n);

/**
   Wait until all the tasks that were already submitted to the context
   \p sched_ctx_id have been terminated.
   See \ref WaitingForTasks for more details.
*/
int starpu_task_wait_for_all_in_ctx(unsigned sched_ctx_id);

/**
   Wait until there are \p n tasks submitted left to be
   executed that were already submitted to the context \p
   sched_ctx_id.
   See \ref WaitingForTasks for more details.
*/
int starpu_task_wait_for_n_submitted_in_ctx(unsigned sched_ctx_id, unsigned n);

/**
   Wait until there is no more ready task.
   See \ref WaitingForTasks for more details.
*/
int starpu_task_wait_for_no_ready(void);

/**
   Return the number of submitted tasks which are ready for execution
   are already executing. It thus does not include tasks waiting for
   dependencies.
   See \ref WaitingForTasks for more details.
*/
int starpu_task_nready(void);

/**
   Return the number of submitted tasks which have not completed yet.
   See \ref WaitingForTasks for more details.
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

   See \ref CreatingAGanttDiagram for more details.
*/
void starpu_iteration_push(unsigned long iteration);

/**
   Drop the iteration number for submitted tasks. This must match a
   previous call to starpu_iteration_push(), and is typically called
   at the end of a task submission loop.
   See \ref CreatingAGanttDiagram for more details.
*/
void starpu_iteration_pop(void);

/**
   See \ref GraphScheduling for more details.
*/
void starpu_do_schedule(void);

/**
   See \ref GraphScheduling for more details.
*/
void starpu_reset_scheduler(void);

/**
   Initialize \p cl with default values. Codelets should preferably be
   initialized statically as shown in \ref DefiningACodelet. However
   such a initialisation is not always possible, e.g. when using C++.
   See \ref DefiningACodelet for more details.
*/
void starpu_codelet_init(struct starpu_codelet *cl);

/**
   Output on \c stderr some statistics on the codelet \p cl.
   See \ref Per-codeletFeedback for more details.
*/
void starpu_codelet_display_stats(struct starpu_codelet *cl);

/**
   Return the task currently executed by the worker, or <c>NULL</c> if
   it is called either from a thread that is not a task or simply
   because there is no task being executed at the moment.
   See \ref Per-taskFeedback for more details.
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

   See \ref SpecifyingATargetNode for more details.
*/
int starpu_task_get_current_data_node(unsigned i);

/**
   Return the name of the performance model of \p task.
   See \ref PerformanceModelExample for more details.
*/
const char *starpu_task_get_model_name(struct starpu_task *task);

/**
   Return the name of \p task, i.e. either its starpu_task::name
   field, or the name of the corresponding performance model.
   See \ref TraceTaskDetails for more details.
*/
const char *starpu_task_get_name(struct starpu_task *task);

/**
   Allocate a task structure which is the exact duplicate of \p task.
   See \ref OtherTaskUtility for more details.
*/
struct starpu_task *starpu_task_dup(struct starpu_task *task);

/**
   This function should be called by schedulers to specify the
   codelet implementation to be executed when executing \p task.
   See \ref SchedulingHelpers for more details.
*/
void starpu_task_set_implementation(struct starpu_task *task, unsigned impl);

/**
   Return the codelet implementation to be executed
   when executing \p task.
   See \ref SchedulingHelpers for more details.
*/
unsigned starpu_task_get_implementation(struct starpu_task *task);

/**
   Create and submit an empty task that unlocks a tag once all its
   dependencies are fulfilled.
   See \ref SynchronizationTasks for more details.
*/
void starpu_create_sync_task(starpu_tag_t sync_tag, unsigned ndeps, starpu_tag_t *deps, void (*callback)(void *), void *callback_arg);

/**
   Create and submit an empty task with the given callback.
   See \ref SynchronizationTasks for more details.
*/
void starpu_create_callback_task(void (*callback)(void *), void *callback_arg);

/**
   Function to be used as a prologue callback to enable fault tolerance for the
   task. This prologue will create a try-task, i.e a duplicate of the task,
   which will to the actual computation.

   The prologue argument can be set to a check_ft function that will be
   called on termination of the duplicate, which can check the result of the
   task, and either confirm success, or resubmit another attempt.
   If it is not set, the default implementation is to just resubmit a new
   try-task.

   See \ref TaskRetry for more details.
*/
void starpu_task_ft_prologue(void *check_ft);

/**
   Create a try-task for a \p meta_task, given a \p template_task task
   template. The meta task can be passed as template on the first call, but
   since it is mangled by starpu_task_ft_create_retry(), further calls
   (typically made by the check_ft callback) need to be passed the previous
   try-task as template task.

   \p check_ft is similar to the prologue argument of
   starpu_task_ft_prologue(), and is typically set to the very function calling
   starpu_task_ft_create_retry().

   The try-task is returned, and can be modified (e.g. to change scheduling
   parameters) before being submitted with starpu_task_submit_nodeps().

   See \ref TaskRetry for more details.
*/
struct starpu_task *starpu_task_ft_create_retry(const struct starpu_task *meta_task, const struct starpu_task *template_task, void (*check_ft)(void *));

/**
   Record that this task failed, and should thus be retried.
   This is usually called from the task codelet function itself, after checking
   the result and noticing that the computation went wrong, and thus the task
   should be retried. The performance of this task execution will not be
   recorded for performance models.

   This can only be called for a task whose data access modes are either
   ::STARPU_R and ::STARPU_W.
*/
void starpu_task_ft_failed(struct starpu_task *task);

/**
   Notify that the try-task was successful and thus the meta-task was
   successful.
   See \ref TaskRetry for more details.
*/
void starpu_task_ft_success(struct starpu_task *meta_task);

/**
   Set the function to call when the watchdog detects that StarPU has
   not finished any task for \ref STARPU_WATCHDOG_TIMEOUT seconds.
   See \ref WatchdogSupport for more details.
*/
void starpu_task_watchdog_set_hook(void (*hook)(void *), void *hook_arg);

/**
   Return the given status as a string
*/
char *starpu_task_status_get_as_string(enum starpu_task_status status);

/**
   Specify a minimum number of submitted tasks allowed at a given
   time, this allows to control the task submission flow. The value
   can also be specified with the environment variable \ref
   STARPU_LIMIT_MIN_SUBMITTED_TASKS.
   See \ref HowToReduceTheMemoryFootprintOfInternalDataStructures for more details.
*/
void starpu_set_limit_min_submitted_tasks(int limit_min);

/**
   Specify a maximum number of submitted tasks allowed at a given
   time, this allows to control the task submission flow. The value
   can also be specified with the environment variable \ref
   STARPU_LIMIT_MAX_SUBMITTED_TASKS.
   See \ref HowToReduceTheMemoryFootprintOfInternalDataStructures for more details.
*/
void starpu_set_limit_max_submitted_tasks(int limit_min);

/** @} */

/**
   @defgroup API_Transactions Transactions
   @{
 */

/**
   Function to open a new transaction object and start the first transaction epoch.

   @return A pointer to an initializes <c>struct starpu_transaction</c>
   or \c NULL if submitting the transaction begin task failed with \c ENODEV.
   See \ref TransactionsCreation for more details.
*/
struct starpu_transaction *starpu_transaction_open(int (*do_start_func)(void *buffer, void *arg), void *do_start_arg);

/**
   Function to mark the end of the current transaction epoch and start a new epoch.
   See \ref TransactionsEpochNext for more details.
*/
void starpu_transaction_next_epoch(struct starpu_transaction *p_trs, void *do_start_arg);

/**
   Function to mark the end of the last transaction epoch and free the transaction object.
   See \ref TransactionsClosing for more details.
*/
void starpu_transaction_close(struct starpu_transaction *p_trs);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_TASK_H__ */
