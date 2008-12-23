#ifndef __PERFMODEL_H__
#define __PERFMODEL_H__

//#include <core/jobs.h>
#include <common/htable32.h>
//#include <core/workers.h>
#include <common/mutex.h>
#include <stdio.h>

struct buffer_descr_t;
struct job_s;
enum archtype;
enum perf_archtype;

/* 
   it is possible that we have multiple versions of the same kind of workers,
   for instance multiple GPUs or even different CPUs within the same machine
   so we do not use the archtype enum type directly for performance models
*/

/* on most system we will consider one or two architectures as all accelerators
   are likely to be identical */
#define NARCH_VARIATIONS	2

enum perf_archtype {
	CORE_DEFAULT = 0,
	CUDA_DEFAULT = 1
};

struct history_entry_t {
	//double measured;
	
	/* mean_n = 1/n sum */
	double mean;

	/* n dev_n = sum2 - 1/n (sum)^2 */
	double deviation;

	/* sum of samples */
	double sum;

	/* sum of samples^2 */
	double sum2;

//	/* sum of ln(measured) */
//	double sumlny;
//
//	/* sum of ln(size) */
//	double sumlnx;
//	double sumlnx2;
//
//	/* sum of ln(size) ln(measured) */
//	double sumlnxlny;
//
	unsigned nsample;

	uint32_t footprint;
	size_t size; /* in bytes */
};

struct history_list_t {
	struct history_list_t *next;
	struct history_entry_t *entry;
};

struct model_list_t {
	struct model_list_t *next;
	struct perfmodel_t *model;
};

struct regression_model_t {
	/* sum of ln(measured) */
	double sumlny;

	/* sum of ln(size) */
	double sumlnx;
	double sumlnx2;

	/* sum of ln(size) ln(measured) */
	double sumlnxlny;

	/* y = alpha size ^ beta */
	double alpha;
	double beta;

	/* y = a size ^b + c */
	double a, b, c;
	unsigned valid;

	unsigned nsample;
};

struct per_arch_perfmodel_t {
	double (*cost_model)(struct buffer_descr_t *t);
	double alpha;
	struct htbl32_node_s *history;
	struct history_list_t *list;
	struct regression_model_t regression;
#ifdef MODEL_DEBUG
	FILE *debug_file;
#endif
};

struct perfmodel_t {
	/* which model is used for that task ? */
	enum {PER_ARCH, COMMON, HISTORY_BASED, REGRESSION_BASED} type;

	/* single cost model */
	double (*cost_model)(struct buffer_descr_t *);

	/* per-architecture model */
	struct per_arch_perfmodel_t per_arch[NARCH_VARIATIONS];
	
	const char *symbol;
	unsigned is_loaded;
	unsigned benchmarking;

	mutex model_mutex;
};

//
///* File format */
//struct model_file_format {
//	unsigned ncore_entries;
//	unsigned ncuda_entries;
//	/* contains core entries, then cuda ones */
//	struct history_entry_t entries[];
//}

double history_based_job_expected_length(struct perfmodel_t *model, enum perf_archtype arch, struct job_s *j);
void register_model(struct perfmodel_t *model);
void dump_registered_models(void);

double job_expected_length(uint32_t who, struct job_s *j, enum perf_archtype arch);
double regression_based_job_expected_length(struct perfmodel_t *model,
		uint32_t who, struct job_s *j);
void update_perfmodel_history(struct job_s *j, enum perf_archtype arch, double measured);

#endif // __PERFMODEL_H__
