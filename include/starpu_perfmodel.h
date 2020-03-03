/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2011       Télécom-SudParis
 * Copyright (C) 2013       Thibaut Lambert
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

#ifndef __STARPU_PERFMODEL_H__
#define __STARPU_PERFMODEL_H__

#include <starpu.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
   @defgroup API_Performance_Model Performance Model
   @{
*/

struct starpu_task;
struct starpu_data_descr;

#define STARPU_NARCH STARPU_ANY_WORKER

/**
   todo
*/
struct starpu_perfmodel_device
{
	enum starpu_worker_archtype type; /**< type of the device */
	int devid;                        /**< identifier of the precise device */
	int ncores;                       /**< number of execution in parallel, minus 1 */
};

/**
   todo
*/
struct starpu_perfmodel_arch
{
	int ndevices;                            /**< number of the devices for the given arch */
	struct starpu_perfmodel_device *devices; /**< list of the devices for the given arch */
};


struct starpu_perfmodel_history_entry
{
	double mean;        /**< mean_n = 1/n sum */
	double deviation;   /**< n dev_n = sum2 - 1/n (sum)^2 */
	double sum;         /**< sum of samples (in µs) */
	double sum2;        /**< sum of samples^2 */
	unsigned nsample;   /**< number of samples */
	unsigned nerror;
	uint32_t footprint; /**< data footprint */
	size_t size;        /**< in bytes */
	double flops;       /**< Provided by the application */

	double duration;
	starpu_tag_t tag;
	double *parameters;
};

struct starpu_perfmodel_history_list
{
	struct starpu_perfmodel_history_list *next;
	struct starpu_perfmodel_history_entry *entry;
};

/**
   todo
*/
struct starpu_perfmodel_regression_model
{
	double sumlny;          /**< sum of ln(measured) */

	double sumlnx;          /**< sum of ln(size) */
	double sumlnx2;         /**< sum of ln(size)^2 */

	unsigned long minx;     /**< minimum size */
	unsigned long maxx;     /**< maximum size */

	double sumlnxlny;       /**< sum of ln(size)*ln(measured) */

	double alpha;           /**< estimated = alpha * size ^ beta */
	double beta;            /**< estimated = alpha * size ^ beta */
	unsigned valid;         /**< whether the linear regression model is valid (i.e. enough measures) */

	double a;               /**< estimated = a size ^b + c */
	double b;               /**< estimated = a size ^b + c */
	double c;               /**< estimated = a size ^b + c */
	unsigned nl_valid;      /**< whether the non-linear regression model is valid (i.e. enough measures) */

	unsigned nsample;       /**< number of sample values for non-linear regression */

	double *coeff;          /**< list of computed coefficients for multiple linear regression model */
	unsigned ncoeff;        /**< number of coefficients for multiple linear regression model */
	unsigned multi_valid;   /**< whether the multiple linear regression model is valid */
};

struct starpu_perfmodel_history_table;

#define starpu_per_arch_perfmodel starpu_perfmodel_per_arch STARPU_DEPRECATED

typedef double (*starpu_perfmodel_per_arch_cost_function)(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl);
typedef size_t (*starpu_perfmodel_per_arch_size_base)(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl);

/**
   information about the performance model of a given arch.
*/
struct starpu_perfmodel_per_arch
{
	/**
	   Used by ::STARPU_PER_ARCH, must point to functions which take a
	   task, the target arch and implementation number (as mere
	   conveniency, since the array is already indexed by these), and
	   must return a task duration estimation in micro-seconds.
	*/
	starpu_perfmodel_per_arch_cost_function cost_function;
	/**
	   Same as in structure starpu_perfmodel, but per-arch, in case it
	   depends on the architecture-specific implementation.
	*/
	starpu_perfmodel_per_arch_size_base size_base;

	/**
	   \private
	   The history of performance measurements.
	*/
	struct starpu_perfmodel_history_table *history;
	/**
	   \private
	   Used by ::STARPU_HISTORY_BASED, ::STARPU_NL_REGRESSION_BASED and
	   ::STARPU_MULTIPLE_REGRESSION_BASED, records all execution history
	   measures.
	*/
	struct starpu_perfmodel_history_list *list;
	/**
	   \private
	   Used by ::STARPU_REGRESSION_BASED, ::STARPU_NL_REGRESSION_BASED
	   and ::STARPU_MULTIPLE_REGRESSION_BASED, contains the estimated
	   factors of the regression.
	*/
	struct starpu_perfmodel_regression_model regression;

	char debug_path[256];
};

/**
   todo
*/
enum starpu_perfmodel_type
{
        STARPU_PERFMODEL_INVALID=0,
	STARPU_PER_ARCH,                  /**< Application-provided per-arch cost model function */
	STARPU_COMMON,                    /**< Application-provided common cost model function, with per-arch factor */
	STARPU_HISTORY_BASED,             /**< Automatic history-based cost model */
	STARPU_REGRESSION_BASED,          /**< Automatic linear regression-based cost model  (alpha * size ^ beta) */
	STARPU_NL_REGRESSION_BASED,       /**< Automatic non-linear regression-based cost model (a * size ^ b + c) */
	STARPU_MULTIPLE_REGRESSION_BASED  /**< Automatic multiple linear regression-based cost model. Application
					     provides parameters, their combinations and exponents. */
};

struct _starpu_perfmodel_state;
typedef struct _starpu_perfmodel_state* starpu_perfmodel_state_t;

/**
   Contain all information about a performance model. At least the
   type and symbol fields have to be filled when defining a performance
   model for a codelet. For compatibility, make sure to initialize the
   whole structure to zero, either by using explicit memset, or by
   letting the compiler implicitly do it in e.g. static storage case. If
   not provided, other fields have to be zero.
*/
struct starpu_perfmodel
{
	/**
	   type of performance model
	   <ul>
	   <li>
	   ::STARPU_HISTORY_BASED, ::STARPU_REGRESSION_BASED,
	   ::STARPU_NL_REGRESSION_BASED: No other fields needs to be
	   provided, this is purely history-based.
	   </li>
	   <li>
	   ::STARPU_MULTIPLE_REGRESSION_BASED: Need to provide fields
	   starpu_perfmodel::nparameters (number of different parameters),
	   starpu_perfmodel::ncombinations (number of parameters
	   combinations-tuples) and table starpu_perfmodel::combinations
	   which defines exponents of the equation. Function cl_perf_func
	   also needs to define how to extract parameters from the task.
	   </li>
	   <li>
	   ::STARPU_PER_ARCH: either field
	   starpu_perfmodel::arch_cost_function has to be filled with a
	   function that returns the cost in micro-seconds on the arch given
	   as parameter, or field starpu_perfmodel::per_arch has to be filled
	   with functions which return the cost in micro-seconds.
	   </li>
	   <li>
	   ::STARPU_COMMON: field starpu_perfmodel::cost_function has to be
	   filled with a function that returns the cost in micro-seconds on a
	   CPU, timing on other archs will be determined by multiplying by an
	   arch-specific factor.
	   </li>
	   </ul>
	*/
	enum starpu_perfmodel_type type;

	/**
	   Used by ::STARPU_COMMON. Take a task and implementation number,
	   and must return a task duration estimation in micro-seconds.
	*/
	double (*cost_function)(struct starpu_task *, unsigned nimpl);
	/**
	   Used by ::STARPU_COMMON. Take a task, an arch and implementation
	   number, and must return a task duration estimation in
	   micro-seconds on that arch.
	*/
	double (*arch_cost_function)(struct starpu_task *, struct starpu_perfmodel_arch * arch, unsigned nimpl);

	/**
	   Used by ::STARPU_HISTORY_BASED, ::STARPU_REGRESSION_BASED and
	   ::STARPU_NL_REGRESSION_BASED. If not <c>NULL</c>, take a task and
	   implementation number, and return the size to be used as index to
	   distinguish histories and as a base for regressions.
	*/
	size_t (*size_base)(struct starpu_task *, unsigned nimpl);
	/**
	   Used by ::STARPU_HISTORY_BASED. If not <c>NULL</c>, take a task
	   and return the footprint to be used as index to distinguish
	   histories. The default is to use the starpu_task_data_footprint()
	   function.
	*/
	uint32_t (*footprint)(struct starpu_task *);

	/**
	   symbol name for the performance model, which will be used as file
	   name to store the model. It must be set otherwise the model will
	   be ignored.
	*/
	const char *symbol;

	/**
	   \private
	   Whether the performance model is already loaded from the disk.
	*/
	unsigned is_loaded;
	/**
	   \private
	*/
	unsigned benchmarking;
	/**
	   \private
	*/
	unsigned is_init;

	void (*parameters)(struct starpu_task * task, double *parameters);
	/**
	   \private
	   Names of parameters used for multiple linear regression models (M,
	   N, K)
	*/
	const char **parameters_names;
	/**
	   \private
	   Number of parameters used for multiple linear regression models
	*/
	unsigned nparameters;
	/**
	   \private
	   Table of combinations of parameters (and the exponents) used for
	   multiple linear regression models
	*/
	unsigned **combinations;
	/**
	   \private
	   Number of combination of parameters used for multiple linear
	   regression models
	*/
	unsigned ncombinations;
	/**
	   \private
	*/
	starpu_perfmodel_state_t state;
};

/**
   Initialize the \p model performance model structure. This is automatically
   called when e.g. submitting a task using a codelet using this performance model.
*/
void starpu_perfmodel_init(struct starpu_perfmodel *model);

/**
   Load the performance model found in the file named \p filename. \p model has to be
   completely zero, and will be filled with the information stored in the given file.
*/
int starpu_perfmodel_load_file(const char *filename, struct starpu_perfmodel *model);

/**
   Load a given performance model. \p model has to be
   completely zero, and will be filled with the information stored in
   <c>$STARPU_HOME/.starpu</c>. The function is intended to be used by
   external tools that want to read the performance model files.
*/

int starpu_perfmodel_load_symbol(const char *symbol, struct starpu_perfmodel *model);

/**
   Unload \p model which has been previously loaded
   through the function starpu_perfmodel_load_symbol()
*/
int starpu_perfmodel_unload_model(struct starpu_perfmodel *model);

/**
  Fills \p path (supposed to be \p maxlen long) with the full path to the
  performance model file for symbol \p symbol.  This path can later on be used
  for instance with starpu_perfmodel_load_file() .
*/
void starpu_perfmodel_get_model_path(const char *symbol, char *path, size_t maxlen);

/**
  Dump performance model \p model to output stream \p output, in XML format.
*/
void starpu_perfmodel_dump_xml(FILE *output, struct starpu_perfmodel *model);

/**
   Free internal memory used for sampling
   management. It should only be called by an application which is not
   calling starpu_shutdown() as this function already calls it. See for
   example <c>tools/starpu_perfmodel_display.c</c>.
*/
void starpu_perfmodel_free_sampling(void);

/**
   Return the architecture type of the worker \p workerid.
*/
struct starpu_perfmodel_arch *starpu_worker_get_perf_archtype(int workerid, unsigned sched_ctx_id);

int starpu_perfmodel_get_narch_combs(void);
int starpu_perfmodel_arch_comb_add(int ndevices, struct starpu_perfmodel_device* devices);
int starpu_perfmodel_arch_comb_get(int ndevices, struct starpu_perfmodel_device *devices);
struct starpu_perfmodel_arch *starpu_perfmodel_arch_comb_fetch(int comb);

struct starpu_perfmodel_per_arch *starpu_perfmodel_get_model_per_arch(struct starpu_perfmodel *model, struct starpu_perfmodel_arch *arch, unsigned impl);
struct starpu_perfmodel_per_arch *starpu_perfmodel_get_model_per_devices(struct starpu_perfmodel *model, int impl, ...);

int starpu_perfmodel_set_per_devices_cost_function(struct starpu_perfmodel *model, int impl, starpu_perfmodel_per_arch_cost_function func, ...);
int starpu_perfmodel_set_per_devices_size_base(struct starpu_perfmodel *model, int impl, starpu_perfmodel_per_arch_size_base func, ...);

/**
   Return the path to the debugging information for the performance model.
*/
void starpu_perfmodel_debugfilepath(struct starpu_perfmodel *model, struct starpu_perfmodel_arch *arch, char *path, size_t maxlen, unsigned nimpl);

char* starpu_perfmodel_get_archtype_name(enum starpu_worker_archtype archtype);

/**
   Return the architecture name for \p arch
*/
void starpu_perfmodel_get_arch_name(struct starpu_perfmodel_arch *arch, char *archname, size_t maxlen, unsigned nimpl);

/**
   Return the estimated time of a task with the given model and the given footprint.
*/
double starpu_perfmodel_history_based_expected_perf(struct starpu_perfmodel *model, struct starpu_perfmodel_arch* arch, uint32_t footprint);

/**
   If starpu_init() is not used, starpu_perfmodel_initialize() should be used called calling starpu_perfmodel_* functions.
*/
void starpu_perfmodel_initialize(void);

/**
   Print a list of all performance models on \p output
*/
int starpu_perfmodel_list(FILE *output);

void starpu_perfmodel_print(struct starpu_perfmodel *model, struct starpu_perfmodel_arch *arch, unsigned nimpl, char *parameter, uint32_t *footprint, FILE *output);
int starpu_perfmodel_print_all(struct starpu_perfmodel *model, char *arch, char *parameter, uint32_t *footprint, FILE *output);
int starpu_perfmodel_print_estimations(struct starpu_perfmodel *model, uint32_t footprint, FILE *output);

int starpu_perfmodel_list_combs(FILE *output, struct starpu_perfmodel *model);

/**
   Feed the performance model model with an explicit
   measurement measured (in µs), in addition to measurements done by StarPU
   itself. This can be useful when the application already has an
   existing set of measurements done in good conditions, that StarPU
   could benefit from instead of doing on-line measurements. An example
   of use can be seen in \ref PerformanceModelExample.
*/
void starpu_perfmodel_update_history(struct starpu_perfmodel *model, struct starpu_task *task, struct starpu_perfmodel_arch *arch, unsigned cpuid, unsigned nimpl, double measured);

/**
   Print the directory name storing performance models on \p output
*/
void starpu_perfmodel_directory(FILE *output);

/**
   Print a matrix of bus bandwidths on \p f.
*/
void starpu_bus_print_bandwidth(FILE *f);

/**
   Print the affinity devices on \p f.
*/
void starpu_bus_print_affinity(FILE *f);

/**
   Print on \p f the name of the files containing the matrix of bus bandwidths, the affinity devices and the latency.
*/
void starpu_bus_print_filenames(FILE *f);

/**
   Return the bandwidth of data transfer between two memory nodes
*/
double starpu_transfer_bandwidth(unsigned src_node, unsigned dst_node);

/**
   Return the latency of data transfer between two memory nodes
*/
double starpu_transfer_latency(unsigned src_node, unsigned dst_node);

/**
   Return the estimated time to transfer a given size between two memory nodes.
*/
double starpu_transfer_predict(unsigned src_node, unsigned dst_node, size_t size);

/**
   Performance model which just always return 1µs.
*/
extern struct starpu_perfmodel starpu_perfmodel_nop;

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_PERFMODEL_H__ */
