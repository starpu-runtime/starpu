#ifndef __PERFMODEL_H__
#define __PERFMODEL_H__

//#include <core/jobs.h>
#include <common/htable32.h>
//#include <core/workers.h>
#include <common/mutex.h>
#include <stdio.h>

struct buffer_descr_t;
struct job_s;

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

	unsigned nsample;
};

struct perfmodel_t {
	/* which model is used for that task ? */
	enum {PER_ARCH, COMMON, HISTORY_BASED} type;

	/* single cost model */
	double (*cost_model)(struct buffer_descr_t *);

	/* per-architecture model */
	double (*cuda_cost_model)(struct buffer_descr_t  *);
	double (*core_cost_model)(struct buffer_descr_t  *);

	/* history-based models */
	struct htbl32_node_s *history_cuda;
	struct htbl32_node_s *history_core;

	struct history_list_t *list_cuda;
	struct history_list_t *list_core;

	struct regression_model_t regression_cuda;
	struct regression_model_t regression_core;
	
	const char *symbol;
	unsigned is_loaded;
	unsigned benchmarking;

	mutex model_mutex;

#ifdef MODEL_DEBUG
	/* for debugging purpose */
	unsigned debug_modelid;
	FILE *cuda_debug_file;
	FILE *core_debug_file;
#endif
};

//
///* File format */
//struct model_file_format {
//	unsigned ncore_entries;
//	unsigned ncuda_entries;
//	/* contains core entries, then cuda ones */
//	struct history_entry_t entries[];
//}

double history_based_job_expected_length(struct perfmodel_t *model, uint32_t who, struct job_s *j);
void dump_registered_models(void);

double job_expected_length(uint32_t who, struct job_s *j);
//void update_perfmodel_history(struct job_s *j, archtype arch, double measured);

#endif // __PERFMODEL_H__
