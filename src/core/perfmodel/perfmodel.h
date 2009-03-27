#ifndef __PERFMODEL_H__
#define __PERFMODEL_H__

#include <common/config.h>
#include <starpu-perfmodel.h>
//#include <core/jobs.h>
#include <common/htable32.h>
//#include <core/workers.h>
#include <starpu-mutex.h>
#include <stdio.h>

struct buffer_descr_t;
struct jobq_s;
struct job_s;
enum archtype;
enum perf_archtype;

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

double data_expected_penalty(struct jobq_s *q, struct job_s *j);

#endif // __PERFMODEL_H__
