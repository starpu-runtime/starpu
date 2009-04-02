/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <unistd.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <core/perfmodel/perfmodel.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <starpu-mutex.h>
#include <datawizard/datawizard.h>
#include <core/perfmodel/regression.h>
#include <common/config.h>

/*
 * History based model
 */


static void insert_history_entry(struct starpu_history_entry_t *entry, struct starpu_history_list_t **list, struct starpu_htbl32_node_s **history_ptr)
{
	struct starpu_history_list_t *link;
	struct starpu_history_entry_t *old;

	link = malloc(sizeof(struct starpu_history_list_t));
	link->next = *list;
	link->entry = entry;
	*list = link;

	old = htbl_insert_32(history_ptr, entry->footprint, entry);
	/* that may fail in case there is some concurrency issue */
	STARPU_ASSERT(old == NULL);
}


static void dump_reg_model(FILE *f, struct starpu_regression_model_t *reg_model)
{
	fprintf(f, "%le\t%le\t%le\t%le\t%le\t%le\t%d\n", reg_model->sumlnx, reg_model->sumlnx2, reg_model->sumlny, reg_model->sumlnxlny, reg_model->alpha, reg_model->beta, reg_model->nsample);
}

static void scan_reg_model(FILE *f, struct starpu_regression_model_t *reg_model)
{
	int res;

	res = fscanf(f, "%le\t%le\t%le\t%le\t%le\t%le\t%d\n", &reg_model->sumlnx, &reg_model->sumlnx2, &reg_model->sumlny, &reg_model->sumlnxlny, &reg_model->alpha, &reg_model->beta, &reg_model->nsample);
	STARPU_ASSERT(res == 7);
}


static void dump_history_entry(FILE *f, struct starpu_history_entry_t *entry)
{
	fprintf(f, "%x\t%zu\t%le\t%le\t%le\t%le\t%d\n", entry->footprint, entry->size, entry->mean, entry->deviation, entry->sum, entry->sum2, entry->nsample);
}

static void scan_history_entry(FILE *f, struct starpu_history_entry_t *entry)
{
	int res;

	res = fscanf(f, "%x\t%zu\t%le\t%le\t%le\t%le\t%d\n", &entry->footprint, &entry->size, &entry->mean, &entry->deviation, &entry->sum, &entry->sum2, &entry->nsample);
	STARPU_ASSERT(res == 7);
}

static void parse_per_arch_model_file(FILE *f, struct starpu_per_arch_perfmodel_t *per_arch_model, unsigned scan_history)
{
	unsigned nentries;

	int res = fscanf(f, "%d\n", &nentries);
	STARPU_ASSERT(res == 1);

	scan_reg_model(f, &per_arch_model->regression);

	res = fscanf(f, "%le\t%le\t%le\n", 
		&per_arch_model->regression.a,
		&per_arch_model->regression.b,
		&per_arch_model->regression.c);
	STARPU_ASSERT(res == 3);

	if (isnan(per_arch_model->regression.a)||isnan(per_arch_model->regression.b)||isnan(per_arch_model->regression.c))
	{
		per_arch_model->regression.valid = 0;
	}
	else {
		per_arch_model->regression.valid = 1;
	}

	if (!scan_history)
		return;

	/* parse core entries */
	unsigned i;
	for (i = 0; i < nentries; i++) {
		struct starpu_history_entry_t *entry = malloc(sizeof(struct starpu_history_entry_t));
		STARPU_ASSERT(entry);

		scan_history_entry(f, entry);
		
		/* insert the entry in the hashtable and the list structures  */
		insert_history_entry(entry, &per_arch_model->list, &per_arch_model->history);
	}
}

static void parse_model_file(FILE *f, struct starpu_perfmodel_t *model, unsigned scan_history)
{
	parse_per_arch_model_file(f, &model->per_arch[STARPU_CORE_DEFAULT], scan_history);
	parse_per_arch_model_file(f, &model->per_arch[STARPU_CUDA_DEFAULT], scan_history);
}

static void dump_per_arch_model_file(FILE *f, struct starpu_per_arch_perfmodel_t *per_arch_model)
{
	/* count the number of elements in the lists */
	struct starpu_history_list_t *ptr;
	unsigned nentries = 0;

	ptr = per_arch_model->list;
	while(ptr) {
		nentries++;
		ptr = ptr->next;
	}

	/* header */
	fprintf(f, "%d\n", nentries);

	dump_reg_model(f, &per_arch_model->regression);

	double a,b,c;
	regression_non_linear_power(per_arch_model->list, &a, &b, &c);
	fprintf(f, "%le\t%le\t%le\n", a, b, c);

	ptr = per_arch_model->list;
	while (ptr) {
		//memcpy(&entries_array[i++], ptr->entry, sizeof(struct starpu_history_entry_t));
		dump_history_entry(f, ptr->entry);
		ptr = ptr->next;
	}
}

static void dump_model_file(FILE *f, struct starpu_perfmodel_t *model)
{
	dump_per_arch_model_file(f, &model->per_arch[STARPU_CORE_DEFAULT]);
	dump_per_arch_model_file(f, &model->per_arch[STARPU_CUDA_DEFAULT]);
}

static void initialize_per_arch_model(struct starpu_per_arch_perfmodel_t *per_arch_model)
{
	per_arch_model->history = NULL;
	per_arch_model->list = NULL;
}

static void initialize_model(struct starpu_perfmodel_t *model)
{
	initialize_per_arch_model(&model->per_arch[STARPU_CORE_DEFAULT]);
	initialize_per_arch_model(&model->per_arch[STARPU_CUDA_DEFAULT]);
}

static struct starpu_model_list_t *registered_models = NULL;
//static unsigned debug_modelid = 0;

#ifdef MODEL_DEBUG
static void get_model_debug_path(struct starpu_perfmodel_t *model, const char *arch, char *path, size_t maxlen)
{
	strncpy(path, PERF_MODEL_DIR, maxlen);
	strncat(path, model->symbol, maxlen);
	
	char hostname[32];
	gethostname(hostname, 32);
	strncat(path, ".", maxlen);
	strncat(path, hostname, maxlen);
	strncat(path, ".", maxlen);
	strncat(path, arch, maxlen);
	strncat(path, ".debug", maxlen);
}
#endif


void register_model(struct starpu_perfmodel_t *model)
{
	/* add the model to a linked list */
	struct starpu_model_list_t *node = malloc(sizeof(struct starpu_model_list_t));

	node->model = model;
	//model->debug_modelid = debug_modelid++;

	/* put this model at the beginning of the list */
	node->next = registered_models;
	registered_models = node;

#ifdef MODEL_DEBUG
	char debugpath[256];
	get_model_debug_path(model, "cuda", debugpath, 256);
	model->per_arch[STARPU_CUDA_DEFAULT].debug_file = fopen(debugpath, "a+");
	STARPU_ASSERT(model->per_arch[STARPU_CUDA_DEFAULT].debug_file);

	get_model_debug_path(model, "core", debugpath, 256);
	model->per_arch[STARPU_CORE_DEFAULT].debug_file = fopen(debugpath, "a+");
	STARPU_ASSERT(model->per_arch[STARPU_CORE_DEFAULT].debug_file);
#endif

	return;
}

static void get_model_path(struct starpu_perfmodel_t *model, char *path, size_t maxlen)
{
	strncpy(path, PERF_MODEL_DIR, maxlen);
	strncat(path, model->symbol, maxlen);
	
	char hostname[32];
	gethostname(hostname, 32);
	strncat(path, ".", maxlen);
	strncat(path, hostname, maxlen);
}

void save_history_based_model(struct starpu_perfmodel_t *model)
{
	STARPU_ASSERT(model);
	STARPU_ASSERT(model->symbol);

	/* TODO checks */

	/* filename = $PERF_MODEL_DIR/symbol.hostname */
	char path[256];
	get_model_path(model, path, 256);

#ifdef VERBOSE
	fprintf(stderr, "Opening performance model file %s for model %s\n", path, model->symbol);
#endif

	/* overwrite existing file, or create it */
	FILE *f;
	f = fopen(path, "w+");
	STARPU_ASSERT(f);

	dump_model_file(f, model);

	fclose(f);

#ifdef DEBUG_MODEL
	fclose(model->cuda_debug_file);
	fclose(model->core_debug_file);
#endif
}

void dump_registered_models(void)
{
	struct starpu_model_list_t *node;
	node = registered_models;

#ifdef VERBOSE
	fprintf(stderr, "DUMP MODELS !\n");
#endif

	while (node) {
		save_history_based_model(node->model);		
		node = node->next;

		/* XXX free node */
	}
}

static int directory_existence_was_tested = 0;

static void create_sampling_directory_if_needed(void)
{
	/* Testing if a directory exists and creating it otherwise 
	   may not be safe: it is possible that the permission are
	   changed in between. Instead, we create it and check if
	   it already existed before */
	int ret;
	ret = mkdir(PERF_MODEL_DIR, S_IRWXU);
	if (ret == -1)
	{
		STARPU_ASSERT(errno == EEXIST);

		/* make sure that it is actually a directory */
		struct stat sb;
		stat(PERF_MODEL_DIR, &sb);
		STARPU_ASSERT(S_ISDIR(sb.st_mode));
	}
}

void load_history_based_model(struct starpu_perfmodel_t *model, unsigned scan_history)
{
	STARPU_ASSERT(model);
	STARPU_ASSERT(model->symbol);

	/* XXX we assume the lock is implicitely initialized (taken = 0) */
	//init_mutex(&model->model_mutex);
	take_mutex(&model->model_mutex);

	/* perhaps some other thread got in before ... */
	if (!model->is_loaded)
	{
		/* make sure the performance model directory exists (or create it) */
		if (!directory_existence_was_tested)
		{
			create_sampling_directory_if_needed();
			directory_existence_was_tested = 1;
		}

		/*
		 * We need to keep track of all the model that were opened so that we can 
		 * possibly update them at runtime termination ...
		 */
		register_model(model);
	
		char path[256];
		get_model_path(model, path, 256);
	
#ifdef VERBOSE
		fprintf(stderr, "Opening performance model file %s for model %s\n", path, model->symbol);
#endif
	
		/* try to open an existing file and load it */
		int res;
		res = access(path, F_OK); 
		if (res == 0) {
		//	fprintf(stderr, "File exists !\n");
	
			FILE *f;
			f = fopen(path, "r");
			STARPU_ASSERT(f);
	
			parse_model_file(f, model, scan_history);
	
			fclose(f);
		}
		else {
			//fprintf(stderr, "File does not exists !\n");
			initialize_model(model);
		}
	
	
		if (starpu_get_env_number("CALIBRATE") != -1)
		{
			fprintf(stderr, "CALIBRATE model %s\n", model->symbol);
			model->benchmarking = 1;
		}
		else {
			model->benchmarking = 0;
		}
	
		model->is_loaded = 1;
	}

	release_mutex(&model->model_mutex);
}

double regression_based_job_expected_length(struct starpu_perfmodel_t *model, enum starpu_perf_archtype arch, struct job_s *j)
{
	double exp = -1.0;
	size_t size = job_get_data_size(j);
	struct starpu_regression_model_t *regmodel;

	if (!model->is_loaded)
		load_history_based_model(model, 0);

	regmodel = &model->per_arch[arch].regression;

	if (regmodel->valid)
		exp = regmodel->a*pow(size, regmodel->b) + regmodel->c;

	return exp;
}

double history_based_job_expected_length(struct starpu_perfmodel_t *model, enum starpu_perf_archtype arch, struct job_s *j)
{
	double exp;
	struct starpu_per_arch_perfmodel_t *per_arch_model;
	struct starpu_history_entry_t *entry;
	struct starpu_htbl32_node_s *history;

	if (!model->is_loaded)
		load_history_based_model(model, 1);

	if (!j->footprint_is_computed)
		compute_buffers_footprint(j);
		
	uint32_t key = j->footprint;

	per_arch_model = &model->per_arch[arch];

	history = per_arch_model->history;
	if (!history)
		return -1.0;

	take_mutex(&model->model_mutex);
	entry = htbl_search_32(history, key);
	release_mutex(&model->model_mutex);

	exp = entry?entry->mean:-1.0;

	return exp;
}

void update_perfmodel_history(job_t j, enum starpu_perf_archtype arch, double measured)
{
	struct starpu_perfmodel_t *model = j->task->cl->model;

	if (model)
	{
		struct starpu_per_arch_perfmodel_t *per_arch_model = &model->per_arch[arch];

		if (model->type == HISTORY_BASED || model->type == REGRESSION_BASED)
		{
			uint32_t key = j->footprint;
			struct starpu_history_entry_t *entry;

			struct starpu_htbl32_node_s *history;
			struct starpu_htbl32_node_s **history_ptr;
			struct starpu_regression_model_t *reg_model;

			struct starpu_history_list_t **list;


			history = per_arch_model->history;
			history_ptr = &per_arch_model->history;
			reg_model = &per_arch_model->regression;
			list = &per_arch_model->list;

			take_mutex(&model->model_mutex);
	
				entry = htbl_search_32(history, key);
	
				if (!entry)
				{
					/* this is the first entry with such a footprint */
					entry = malloc(sizeof(struct starpu_history_entry_t));
					STARPU_ASSERT(entry);
						entry->mean = measured;
						entry->sum = measured;
	
						entry->deviation = 0.0;
						entry->sum2 = measured*measured;
	
						entry->size = job_get_data_size(j);
	
						entry->footprint = key;
						entry->nsample = 1;
	
					insert_history_entry(entry, list, history_ptr);
	
				}
				else {
					/* there is already some entry with the same footprint */
					entry->sum += measured;
					entry->sum2 += measured*measured;
					entry->nsample++;
	
					unsigned n = entry->nsample;
					entry->mean = entry->sum / n;
					entry->deviation = sqrt((entry->sum2 - (entry->sum*entry->sum)/n)/n);
				}
			
				STARPU_ASSERT(entry);
			
			/* update the regression model as well */
			double logy, logx;
			logx = logl(entry->size);
			logy = logl(measured);

			reg_model->sumlnx += logx;
			reg_model->sumlnx2 += logx*logx;
			reg_model->sumlny += logy;
			reg_model->sumlnxlny += logx*logy;
			reg_model->nsample++;

			unsigned n = reg_model->nsample;
			
			double num = (n*reg_model->sumlnxlny - reg_model->sumlnx*reg_model->sumlny);
			double denom = (n*reg_model->sumlnx2 - reg_model->sumlnx*reg_model->sumlnx);

			reg_model->beta = num/denom;
			reg_model->alpha = expl((reg_model->sumlny - reg_model->beta*reg_model->sumlnx)/n);
			
			release_mutex(&model->model_mutex);
		}

#ifdef MODEL_DEBUG
		FILE * debug_file = per_arch_model->debug_file;

		take_mutex(&model->model_mutex);

		fprintf(debug_file, "%lf\t", measured);
		unsigned i;
			
		for (i = 0; i < j->nbuffers; i++)
		{
			data_state *state = j->buffers[i].state;

			STARPU_ASSERT(state->ops);
			STARPU_ASSERT(state->ops->display);
			state->ops->display(state, debug_file);
		}
		fprintf(debug_file, "\n");	


		release_mutex(&model->model_mutex);
#endif
	}
}
