#ifndef __PERFMODEL_H__
#define __PERFMODEL_H__

//#include <core/jobs.h>
#include <common/htable32.h>
//#include <core/workers.h>
#include <stdio.h>

struct buffer_descr_t;
struct job_s;

struct history_entry_t {
	double measured;
	uint32_t footprint;
	unsigned nsample;
};

struct history_list_t {
	struct history_list_t *next;
	struct history_entry_t *entry;
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
	
	const char *model_symbol;
	FILE *model_file;
	unsigned is_loaded;
	unsigned benchmarking;
};

double job_expected_length(uint32_t who, struct job_s *j);
//void update_perfmodel_history(struct job_s *j, archtype arch, double measured);

#endif // __PERFMODEL_H__
