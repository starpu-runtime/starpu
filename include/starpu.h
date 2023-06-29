/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2021       Federal University of Rio Grande do Sul (UFRGS)
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

#ifndef __STARPU_H__
#define __STARPU_H__

#include <stdlib.h>

#ifndef _MSC_VER
#include <stdint.h>
#else
#include <windows.h>
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef UINT_PTR uintptr_t;
typedef char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long long int64_t;
typedef INT_PTR intptr_t;
#endif

#include <starpu_config.h>

#ifdef STARPU_HAVE_WINDOWS
#include <windows.h>
#endif

#if defined(STARPU_USE_OPENCL) && !defined(__CUDACC__)
#include <starpu_opencl.h>
#endif

#include <starpu_thread.h>
#include <starpu_thread_util.h>
#include <starpu_util.h>
#include <starpu_data.h>
#include <starpu_helper.h>
#include <starpu_disk.h>
#include <starpu_data_interfaces.h>
#include <starpu_data_filters.h>
#include <starpu_stdlib.h>
#include <starpu_task_bundle.h>
#include <starpu_task_dep.h>
#include <starpu_task.h>
#include <starpu_worker.h>
#include <starpu_perfmodel.h>
#include <starpu_worker.h>
#ifndef BUILDING_STARPU
#include <starpu_task_list.h>
#endif
#include <starpu_task_util.h>
#include <starpu_scheduler.h>
#include <starpu_sched_ctx.h>
#include <starpu_expert.h>
#include <starpu_rand.h>
#include <starpu_cuda.h>
#include <starpu_hip.h>
#include <starpu_hipblas.h>
#include <starpu_cublas.h>
#include <starpu_cusparse.h>
#include <starpu_bound.h>
#include <starpu_hash.h>
#include <starpu_profiling.h>
#include <starpu_profiling_tool.h>
#include <starpu_fxt.h>
#include <starpu_driver.h>
#include <starpu_tree.h>
#include <starpu_openmp.h>
#include <starpu_simgrid_wrap.h>
#include <starpu_bitmap.h>
#include <starpu_parallel_worker.h>
#include <starpu_perf_monitoring.h>
#include <starpu_perf_steering.h>
#include <starpu_max_fpga.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
   @defgroup API_Initialization_and_Termination Initialization and Termination
   @{
*/

/**
   Structure passed to the starpu_init() function to configure StarPU.
   It has to be initialized with starpu_conf_init(). When the default
   value is used, StarPU automatically selects the number of
   processing units and takes the default scheduling policy. The
   environment variables overwrite the equivalent parameters unless
   starpu_conf::precedence_over_environment_variables is set.
*/
struct starpu_conf
{
	/**
	   @private
	   Will be initialized by starpu_conf_init(). Should not be
	   set by hand.
	*/
	int magic;

	/**
	   @private
	   Tell starpu_init() if MPI will be initialized later.
	*/
	int will_use_mpi;

	/**
	   Name of the scheduling policy. This can also be specified
	   with the environment variable \ref STARPU_SCHED. (default =
	   <c>NULL</c>).
	*/
	const char *sched_policy_name;

	/**
	   Definition of the scheduling policy. This field is ignored
	   if starpu_conf::sched_policy_name is set.
	   (default = <c>NULL</c>)
	*/
	struct starpu_sched_policy *sched_policy;

	/**
	   Callback function that can later be used by the scheduler.
	   The scheduler can retrieve this function by calling
	   starpu_sched_ctx_get_sched_policy_callback()
	*/
	void (*sched_policy_callback)(unsigned);

	/**
	   For all parameters specified in this structure that can
	   also be set with environment variables, by default,
	   StarPU chooses the value of the environment variable
	   against the value set in starpu_conf. Setting the parameter
	   starpu_conf::precedence_over_environment_variables to 1 allows to give precedence
	   to the value set in the structure over the environment
	   variable.
	 */
	int precedence_over_environment_variables;

	/**
	   Number of CPU cores that StarPU can use. This can also be
	   specified with the environment variable \ref STARPU_NCPU.
	   (default = \c -1)
	*/
	int ncpus;

	/**
	   Number of CPU cores to that StarPU should leave aside. They can then
	   be used by application threads, by calling starpu_get_next_bindid() to
	   get their ID, and starpu_bind_thread_on() to bind the current thread to them.
	  */
	int reserve_ncpus;

	/**
	   Number of CUDA devices that StarPU can use. This can also
	   be specified with the environment variable \ref
	   STARPU_NCUDA.
	   (default = \c -1)
	*/
	int ncuda;

	/**
	   Number of HIP devices that StarPU can use. This can also
	   be specified with the environment variable \ref
	   STARPU_NHIP.
	   (default = \c -1)
	*/
	int nhip;

	/**
	   Number of OpenCL devices that StarPU can use. This can also
	   be specified with the environment variable \ref
	   STARPU_NOPENCL.
	   (default = \c -1)
	*/
	int nopencl;

	/**
	   Number of Maxeler FPGA devices that StarPU can use. This can also
	   be specified with the environment variable \ref
	   STARPU_NMAX_FPGA.
	   (default = -1)
	*/
	int nmax_fpga;

	/**
	   Number of MPI Master Slave devices that StarPU can use.
	   This can also be specified with the environment variable
	   \ref STARPU_NMPI_MS.
	   (default = \c -1)
	*/
	int nmpi_ms;

	/**
	   Number of TCP/IP Master Slave devices that StarPU can use.
	   This can also be specified with the environment variable
	   \ref STARPU_NTCPIP_MS.
	   (default = \c -1)
	*/
	int ntcpip_ms;

	/**
	   If this flag is set, the starpu_conf::workers_bindid array
	   indicates where the different workers are bound, otherwise
	   StarPU automatically selects where to bind the different
	   workers. This can also be specified with the environment
	   variable \ref STARPU_WORKERS_CPUID.
	   (default = \c 0)
	*/
	unsigned use_explicit_workers_bindid;

	/**
	   If the starpu_conf::use_explicit_workers_bindid flag is
	   set, this array indicates where to bind the different
	   workers. The i-th entry of the starpu_conf::workers_bindid
	   indicates the logical identifier of the processor which
	   should execute the i-th worker. Note that the logical
	   ordering of the CPUs is either determined by the OS, or
	   provided by the \c hwloc library in case it is available.
	*/
	unsigned workers_bindid[STARPU_NMAXWORKERS];

	/**
	   If this flag is set, the CUDA workers will be attached to
	   the CUDA devices specified in the
	   starpu_conf::workers_cuda_gpuid array. Otherwise, StarPU
	   affects the CUDA devices in a round-robin fashion. This can
	   also be specified with the environment variable \ref
	   STARPU_WORKERS_CUDAID.
	   (default = \c 0)
	*/
	unsigned use_explicit_workers_cuda_gpuid;

	/**
	   If the starpu_conf::use_explicit_workers_cuda_gpuid flag is
	   set, this array contains the logical identifiers of the
	   CUDA devices (as used by \c cudaGetDevice()).
	*/
	unsigned workers_cuda_gpuid[STARPU_NMAXWORKERS];

	/**
	   If this flag is set, the HIP workers will be attached to
	   the HIP devices specified in the
	   starpu_conf::workers_hip_gpuid array. Otherwise, StarPU
	   affects the HIP devices in a round-robin fashion. This can
	   also be specified with the environment variable \ref
	   STARPU_WORKERS_HIPID.
	   (default = \c 0)
	*/
	unsigned use_explicit_workers_hip_gpuid;

	/**
	   If the starpu_conf::use_explicit_workers_hip_gpuid flag is
	   set, this array contains the logical identifiers of the
	   HIP devices (as used by \c hipGetDevice()).
	*/
	unsigned workers_hip_gpuid[STARPU_NMAXWORKERS];

	/**
	   If this flag is set, the OpenCL workers will be attached to
	   the OpenCL devices specified in the
	   starpu_conf::workers_opencl_gpuid array. Otherwise, StarPU
	   affects the OpenCL devices in a round-robin fashion. This
	   can also be specified with the environment variable \ref
	   STARPU_WORKERS_OPENCLID.
	   (default = \c 0)
	*/
	unsigned use_explicit_workers_opencl_gpuid;

	/**
	   If the starpu_conf::use_explicit_workers_opencl_gpuid flag
	   is set, this array contains the logical identifiers of the
	   OpenCL devices to be used.
	*/
	unsigned workers_opencl_gpuid[STARPU_NMAXWORKERS];

	/**
	   If this flag is set, the Maxeler FPGA workers will be attached to
	   the Maxeler FPGA devices specified in the
	   starpu_conf::workers_max_fpga_deviceid array. Otherwise, StarPU
	   affects the Maxeler FPGA devices in a round-robin fashion. This
	   can also be specified with the environment variable \ref
	   STARPU_WORKERS_MAX_FPGAID.
	   (default = 0)
	*/
	unsigned use_explicit_workers_max_fpga_deviceid;

	/**
	   If the starpu_conf::use_explicit_workers_max_fpga_deviceid flag
	   is set, this array contains the logical identifiers of the
	   Maxeler FPGA devices to be used.
	*/
	unsigned workers_max_fpga_deviceid[STARPU_NMAXWORKERS];

#ifdef STARPU_USE_MAX_FPGA
	/**
           This allows to specify the Maxeler file(s) to be loaded on Maxeler FPGAs.
	   This is an array of starpu_max_load, the last of which shall have
	   file set to NULL. In order to use all available devices,
	   starpu_max_load::engine_id_pattern can be set to "*", but only the
           last non-NULL entry can be set so.

	   If this is not set, it is assumed that the basic static SLiC
           interface is used.
        */
	struct starpu_max_load *max_fpga_load;
#else
	void *max_fpga_load;
#endif

	/**
	   If this flag is set, the MPI Master Slave workers will be
	   attached to the MPI Master Slave devices specified in the
	   array starpu_conf::workers_mpi_ms_deviceid. Otherwise,
	   StarPU affects the MPI Master Slave devices in a
	   round-robin fashion.
	   (default = \c 0)
	*/
	unsigned use_explicit_workers_mpi_ms_deviceid;

	/**
	   If the flag
	   starpu_conf::use_explicit_workers_mpi_ms_deviceid is set,
	   the array contains the logical identifiers of the MPI
	   Master Slave devices to be used.
	*/
	unsigned workers_mpi_ms_deviceid[STARPU_NMAXWORKERS];

	/**
	   If this flag is set, StarPU will recalibrate the bus.  If
	   this value is equal to -1, the default value is used. This
	   can also be specified with the environment variable \ref
	   STARPU_BUS_CALIBRATE.
	   (default = \c 0)
	*/
	int bus_calibrate;

	/**
	   If this flag is set, StarPU will calibrate the performance
	   models when executing tasks. If this value is equal to -1,
	   the default value is used. If the value is equal to 1, it
	   will force continuing calibration. If the value is equal to
	   2, the existing performance models will be overwritten.
	   This can also be specified with the environment variable
	   \ref STARPU_CALIBRATE.
	   (default = \c 0)
	*/
	int calibrate;

	/**
	   This flag should be set to 1 to enforce data locality when
	   choosing a worker to execute a task.
	   This can also be specified with the environment variable
	   \ref STARPU_DATA_LOCALITY_ENFORCE.
	   This can also be specified at compilation time by giving to
	   the configure script the option \ref
	   enable-data-locality-enforce "--enable-data-locality-enforce".
	   (default = \c 0)
	*/
	int data_locality_enforce;

	/**
	   By default, StarPU executes parallel tasks concurrently.
	   Some parallel libraries (e.g. most OpenMP implementations)
	   however do not support concurrent calls to parallel code.
	   In such case, setting this flag makes StarPU only start one
	   parallel task at a time (but other CPU and GPU tasks are
	   not affected and can be run concurrently). The parallel
	   task scheduler will however still try varying combined
	   worker sizes to look for the most efficient ones.
	   This can also be specified with the environment variable
	   \ref STARPU_SINGLE_COMBINED_WORKER.
	   (default = \c 0)
	*/
	int single_combined_worker;

	/**
	   This flag should be set to 1 to disable asynchronous copies
	   between CPUs and all accelerators.
	   The AMD implementation of OpenCL is known to fail when
	   copying data asynchronously. When using this
	   implementation, it is therefore necessary to disable
	   asynchronous data transfers.
	   This can also be specified with the environment variable
	   \ref STARPU_DISABLE_ASYNCHRONOUS_COPY.
	   This can also be specified at compilation time by giving to
	   the configure script the option \ref
	   disable-asynchronous-copy "--disable-asynchronous-copy".
	   (default = \c 0)
	*/
	int disable_asynchronous_copy;

	/**
	   This flag should be set to 1 to disable asynchronous copies
	   between CPUs and CUDA accelerators.
	   This can also be specified with the environment variable
	   \ref STARPU_DISABLE_ASYNCHRONOUS_CUDA_COPY.
	   This can also be specified at compilation time by giving to
	   the configure script the option \ref
	   disable-asynchronous-cuda-copy
	   "--disable-asynchronous-cuda-copy".
	   (default = \c 0)
	*/
	int disable_asynchronous_cuda_copy;

	/**
	   This flag should be set to 1 to disable asynchronous copies
	   between CPUs and HIP accelerators.
	   This can also be specified with the environment variable
	   \ref STARPU_DISABLE_ASYNCHRONOUS_HIP_COPY.
	   This can also be specified at compilation time by giving to
	   the configure script the option \ref
	   disable-asynchronous-hip-copy
	   "--disable-asynchronous-hip-copy".
	   (default = \c 0)
	*/
	int disable_asynchronous_hip_copy;

	/**
	   This flag should be set to 1 to disable asynchronous copies
	   between CPUs and OpenCL accelerators.
	   The AMD implementation of OpenCL is known to fail when
	   copying data asynchronously. When using this
	   implementation, it is therefore necessary to disable
	   asynchronous data transfers.
	   This can also be specified with the environment variable
	   \ref STARPU_DISABLE_ASYNCHRONOUS_OPENCL_COPY.
	   This can also be specified at compilation time by giving to
	   the configure script the option \ref
	   disable-asynchronous-opencl-copy
	   "--disable-asynchronous-opencl-copy".
	   (default = \c 0)
	*/
	int disable_asynchronous_opencl_copy;

	/**
	   This flag should be set to 1 to disable asynchronous copies
	   between CPUs and MPI Master Slave devices.
	   This can also be specified with the environment variable
	   \ref STARPU_DISABLE_ASYNCHRONOUS_MPI_MS_COPY.
	   This can also be specified at compilation time by giving to
	   the configure script the option \ref
	   disable-asynchronous-mpi-master-slave-copy
	   "--disable-asynchronous-mpi-master-slave-copy".
	   (default = \c 0).
	*/
	int disable_asynchronous_mpi_ms_copy;

	/**
	   This flag should be set to 1 to disable asynchronous copies
	   between CPUs and TCP/IP Master Slave devices.
	   This can also be specified with the environment variable
	   \ref STARPU_DISABLE_ASYNCHRONOUS_TCPIP_MS_COPY.
	   This can also be specified at compilation time by giving to
	   the configure script the option \ref
	   disable-asynchronous-tcpip-master-slave-copy
	   "--disable-asynchronous-tcpip-master-slave-copy".
	   (default = \c 0).
	*/
	int disable_asynchronous_tcpip_ms_copy;

	/**
	   This flag should be set to 1 to disable asynchronous copies
	   between CPUs and Maxeler FPGA devices.
	   This can also be specified with the environment variable
	   \ref STARPU_DISABLE_ASYNCHRONOUS_MAX_FPGA_COPY.
	   This can also be specified at compilation time by giving to
	   the configure script the option \ref
	   disable-asynchronous-fpga-copy
	   "--disable-asynchronous-fpga-copy".
	   (default = 0).
	*/
	int disable_asynchronous_max_fpga_copy;

	/**
	   This flag should be set to 1 to disable memory mapping
	   support between memory nodes. This can also be specified
	   with the environment variable \ref STARPU_ENABLE_MAP.
	*/
	int enable_map;

	/**
	   Enable CUDA/OpenGL interoperation on these CUDA devices.
	   This can be set to an array of CUDA device identifiers for
	   which \c cudaGLSetGLDevice() should be called instead of \c
	   cudaSetDevice(). Its size is specified by the
	   starpu_conf::n_cuda_opengl_interoperability field below
	   (default = <c>NULL</c>)
	*/
	unsigned *cuda_opengl_interoperability;

	/**
	   Size of the array starpu_conf::cuda_opengl_interoperability
	*/
	unsigned n_cuda_opengl_interoperability;

	/**
	   Array of drivers that should not be launched by StarPU. The
	   application will run in one of its own threads.
	   (default = <c>NULL</c>)
	*/
	struct starpu_driver *not_launched_drivers;

	/**
	   The number of StarPU drivers that should not be launched by
	   StarPU, i.e number of elements of the array
	   starpu_conf::not_launched_drivers.
	   (default = \c 0)
	*/
	unsigned n_not_launched_drivers;

	/**
	   Specify the buffer size used for FxT tracing. Starting from
	   FxT version 0.2.12, the buffer will automatically be
	   flushed when it fills in, but it may still be interesting
	   to specify a bigger value to avoid any flushing (which
	   would disturb the trace).
	*/
	uint64_t trace_buffer_size;

	/**
	   Set the minimum priority used by priorities-aware
	   schedulers.
	   This also can be specified with the environment variable \ref
	   STARPU_MIN_PRIO
	*/
	int global_sched_ctx_min_priority;

	/**
	   Set the maximum priority used by priorities-aware
	   schedulers.
	   This also can be specified with the environment variable \ref
	   STARPU_MAX_PRIO
	*/
	int global_sched_ctx_max_priority;

#ifdef STARPU_WORKER_CALLBACKS
	void (*callback_worker_going_to_sleep)(unsigned workerid);
	void (*callback_worker_waking_up)(unsigned workerid);
#endif

	/**
	   Specify if StarPU should catch \c SIGINT, \c SIGSEGV and \c SIGTRAP
	   signals to make sure final actions (e.g dumping FxT trace
	   files) are done even though the application has crashed. By
	   default (value = \c 1), signals are caught. It should be
	   disabled on systems which already catch these signals for
	   their own needs (e.g JVM)
	   This can also be specified with the environment variable
	   \ref STARPU_CATCH_SIGNALS.
	 */
	int catch_signals;

	/**
	   Specify whether StarPU should automatically start to collect
	   performance counters after initialization
	 */
	unsigned start_perf_counter_collection;

	/**
	   Minimum spinning backoff of drivers (default = \c 1)
	 */
	unsigned driver_spinning_backoff_min;

	/**
	   Maximum spinning backoff of drivers. (default = \c 32)
	 */
	unsigned driver_spinning_backoff_max;

	/**
	   Specify if CUDA workers should do only fast allocations
	   when running the datawizard progress of
	   other memory nodes. This will pass the interval value
	   _STARPU_DATAWIZARD_ONLY_FAST_ALLOC to the allocation method.
	   Default value is 0, allowing CUDA workers to do slow
	   allocations.
	   This can also be specified with the environment variable
	   \ref STARPU_CUDA_ONLY_FAST_ALLOC_OTHER_MEMNODES.
	 */
	int cuda_only_fast_alloc_other_memnodes;
};

/**
   Initialize the \p conf structure with the default values. In case
   some configuration parameters are already specified through
   environment variables, starpu_conf_init() initializes the fields of
   \p conf according to the environment variables.
   For instance if \ref STARPU_CALIBRATE is set, its value is put in
   the field starpu_conf::calibrate of \p conf.
   Upon successful completion, this function returns 0. Otherwise,
   <c>-EINVAL</c> indicates that the argument was <c>NULL</c>.
*/
int starpu_conf_init(struct starpu_conf *conf);

/**
   Set fields of \p conf so that no worker is enabled, i.e. set
   starpu_conf::ncpus = 0, starpu_conf::ncuda = 0, etc.

   This allows to portably enable only a given type of worker:
   <br/>
   <c>
   starpu_conf_noworker(&conf);<br/>
   conf.ncpus = -1;
   </c>

   See \ref ConfigurationAndInitialization for more details.
*/
int starpu_conf_noworker(struct starpu_conf *conf);

/**
   StarPU initialization method, must be called prior to any other
   StarPU call. It is possible to specify StarPU’s configuration (e.g.
   scheduling policy, number of cores, ...) by passing a
   non-<c>NULL</c> \p conf. Default configuration is used if \p conf
   is <c>NULL</c>. Upon successful completion, this function returns
   0. Otherwise, <c>-ENODEV</c> indicates that no worker was available
   (and thus StarPU was not initialized). See \ref SubmittingATask for more details.
*/
int starpu_init(struct starpu_conf *conf) STARPU_WARN_UNUSED_RESULT;

/**
   Similar to starpu_init(), but also take the \p argc and \p argv as
   defined by the application, which is necessary when running in
   Simgrid mode or MPI Master Slave mode.
   Do not call starpu_init() and starpu_initialize() in the same
   program. See \ref SubmittingATask for more details.
*/
int starpu_initialize(struct starpu_conf *user_conf, int *argc, char ***argv);

/**
   Return 1 if StarPU is already initialized. See \ref ConfigurationAndInitialization for more details.
*/
int starpu_is_initialized(void);

/**
   Wait for starpu_init() call to finish. See \ref ConfigurationAndInitialization for more details.
*/
void starpu_wait_initialized(void);

/**
   StarPU termination method, must be called at the end of the
   application: statistics and other post-mortem debugging information
   are not guaranteed to be available until this method has been
   called. See \ref SubmittingATask for more details.
*/
void starpu_shutdown(void);

/**
   Suspend the processing of new tasks by workers. It can be used in a
   program where StarPU is used during only a part of the execution.
   Without this call, the workers continue to poll for new tasks in a
   tight loop, wasting CPU time. The symmetric call to starpu_resume()
   should be used to unfreeze the workers. See \ref KernelThreadsStartedByStarPU and \ref PauseResume for more details.
*/
void starpu_pause(void);

/**
   Symmetrical call to starpu_pause(), used to resume the workers
   polling for new tasks. This would be typically called only once
   having submitted all tasks. See \ref KernelThreadsStartedByStarPU and \ref PauseResume for more details.
*/
void starpu_resume(void);

/**
   Return !0 if task processing by workers is currently paused, 0 otherwise.
   See \ref StarPUEatsCPUs for more details.
 */
int starpu_is_paused(void);

/**
   Value to be passed to starpu_get_next_bindid() and
   starpu_bind_thread_on() when binding a thread which will
   significantly eat CPU time, and should thus have its own dedicated
   CPU.
*/
#define STARPU_THREAD_ACTIVE (1 << 0)

/**
   Return a PU binding ID which can be used to bind threads with
   starpu_bind_thread_on(). \p flags can be set to
   ::STARPU_THREAD_ACTIVE or 0. When \p npreferred is set to non-zero,
   \p preferred is an array of size \p npreferred in which a
   preference of PU binding IDs can be set. By default StarPU will
   return the first PU available for binding. 
   See \ref KernelThreadsStartedByStarPU and \ref cpuWorkers for more details.
*/
unsigned starpu_get_next_bindid(unsigned flags, unsigned *preferred, unsigned npreferred);

/**
   Bind the calling thread on the given \p cpuid (which should have
   been obtained with starpu_get_next_bindid()).

   Return -1 if a thread was already bound to this PU (but binding
   will still have been done, and a warning will have been printed),
   so the caller can tell the user how to avoid the issue.

   \p name should be set to a unique string so that different calls
   with the same name for the same \p cpuid does not produce a warning.

   See \ref KernelThreadsStartedByStarPU and \ref cpuWorkers for more details.
*/
int starpu_bind_thread_on(int cpuid, unsigned flags, const char *name);

/**
   Print a description of the topology on \p f.
   See \ref ConfigurationAndInitialization for more details.
*/
void starpu_topology_print(FILE *f);

/**
   Return 1 if asynchronous data transfers between CPU and
   accelerators are disabled.
   See \ref Basic for more details.
*/
int starpu_asynchronous_copy_disabled(void);

/**
   Return 1 if asynchronous data transfers between CPU and CUDA
   accelerators are disabled.
   See \ref cudaWorkers for more details.
*/
int starpu_asynchronous_cuda_copy_disabled(void);

/**
   Return 1 if asynchronous data transfers between CPU and HIP
   accelerators are disabled.
   See \ref hipWorkers for more details.
*/
int starpu_asynchronous_hip_copy_disabled(void);

/**
   Return 1 if asynchronous data transfers between CPU and OpenCL
   accelerators are disabled.
   See \ref openclWorkers for more details.
*/
int starpu_asynchronous_opencl_copy_disabled(void);

/**
   Return 1 if asynchronous data transfers between CPU and Maxeler FPGA
   devices are disabled.
   See \ref maxfpgaWorkers for more details.
*/
int starpu_asynchronous_max_fpga_copy_disabled(void);

/**
   Return 1 if asynchronous data transfers between CPU and MPI Slave
   devices are disabled.
   See \ref mpimsWorkers for more details.
*/
int starpu_asynchronous_mpi_ms_copy_disabled(void);

/**
   Return 1 if asynchronous data transfers between CPU and TCP/IP Slave
   devices are disabled.
   See \ref tcpipmsWorkers for more details.
*/
int starpu_asynchronous_tcpip_ms_copy_disabled(void);

/**
   Return 1 if asynchronous data transfers with a given kind of memory
   are disabled.
*/
int starpu_asynchronous_copy_disabled_for(enum starpu_node_kind kind);

/**
   Return 1 if memory mapping support between memory nodes is
   enabled.
   See \ref Basic for more details.
*/
int starpu_map_enabled(void);

/**
   Call starpu_profiling_bus_helper_display_summary() and
   starpu_profiling_worker_helper_display_summary().
   See \ref DataStatistics for more details.
*/
void starpu_display_stats(void);

/** @} */

/**
   @defgroup API_Versioning Versioning
   @{
*/

/**
   Return as 3 integers the version of StarPU used when running the
   application.
   See \ref ConfigurationAndInitialization for more details.
*/
void starpu_get_version(int *major, int *minor, int *release);

/** @} */

#ifdef __cplusplus
}
#endif

#include "starpu_deprecated_api.h"

#endif /* __STARPU_H__ */
