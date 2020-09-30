/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#if !defined(_WIN32) || defined(__MINGW32__) || defined(__CYGWIN__)
#include <dirent.h>
#include <sys/stat.h>
#endif
#include <errno.h>
#include <common/config.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#include <common/utils.h>
#include <core/perfmodel/perfmodel.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <datawizard/datawizard.h>
#include <core/perfmodel/regression.h>
#include <core/perfmodel/multiple_regression.h>
#include <common/config.h>
#include <starpu_parameters.h>
#include <common/uthash.h>
#include <limits.h>
#include <core/task.h>

#ifdef STARPU_HAVE_WINDOWS
#include <windows.h>
#endif

#define HASH_ADD_UINT32_T(head,field,add) HASH_ADD(hh,head,field,sizeof(uint32_t),add)
#define HASH_FIND_UINT32_T(head,find,out) HASH_FIND(hh,head,find,sizeof(uint32_t),out)

#define STR_SHORT_LENGTH 32
#define STR_LONG_LENGTH 256
#define STR_VERY_LONG_LENGTH 1024

static struct starpu_perfmodel_arch **arch_combs;
static int current_arch_comb;
static int nb_arch_combs;
static starpu_pthread_rwlock_t arch_combs_mutex;
static int historymaxerror;
static char ignore_devid[STARPU_ANY_WORKER];

/* How many executions a codelet will have to be measured before we
 * consider that calibration will provide a value good enough for scheduling */
unsigned _starpu_calibration_minimum;

struct starpu_perfmodel_history_table
{
	UT_hash_handle hh;
	uint32_t footprint;
	struct starpu_perfmodel_history_entry *history_entry;
};

/* We want more than 10% variance on X to trust regression */
#define VALID_REGRESSION(reg_model) \
	((reg_model)->minx < (9*(reg_model)->maxx)/10 && (reg_model)->nsample >= _starpu_calibration_minimum)

static starpu_pthread_rwlock_t registered_models_rwlock;
LIST_TYPE(_starpu_perfmodel,
	struct starpu_perfmodel *model;
)
static struct _starpu_perfmodel_list registered_models;

void _starpu_perfmodel_malloc_per_arch(struct starpu_perfmodel *model, int comb, int nb_impl)
{
	int i;

	_STARPU_MALLOC(model->state->per_arch[comb], nb_impl*sizeof(struct starpu_perfmodel_per_arch));
	for(i = 0; i < nb_impl; i++)
	{
		memset(&model->state->per_arch[comb][i], 0, sizeof(struct starpu_perfmodel_per_arch));
	}
	model->state->nimpls_set[comb] = nb_impl;
}

void _starpu_perfmodel_malloc_per_arch_is_set(struct starpu_perfmodel *model, int comb, int nb_impl)
{
	int i;

	_STARPU_MALLOC(model->state->per_arch_is_set[comb], nb_impl*sizeof(int));
	for(i = 0; i < nb_impl; i++)
	{
		model->state->per_arch_is_set[comb][i] = 0;
	}
}

int _starpu_perfmodel_arch_comb_get(int ndevices, struct starpu_perfmodel_device *devices)
{
	int comb, ncomb;
	ncomb = current_arch_comb;
	for(comb = 0; comb < ncomb; comb++)
	{
		int found = 0;
		if(arch_combs[comb]->ndevices == ndevices)
		{
			int dev1, dev2;
			int nfounded = 0;
			for(dev1 = 0; dev1 < arch_combs[comb]->ndevices; dev1++)
			{
				for(dev2 = 0; dev2 < ndevices; dev2++)
				{
					if(arch_combs[comb]->devices[dev1].type == devices[dev2].type &&
					   (ignore_devid[devices[dev2].type] ||
					    arch_combs[comb]->devices[dev1].devid == devices[dev2].devid) &&
					   arch_combs[comb]->devices[dev1].ncores == devices[dev2].ncores)
						nfounded++;
				}
			}
			if(nfounded == ndevices)
				found = 1;
		}
		if (found)
			return comb;
	}
	return -1;
}

int starpu_perfmodel_arch_comb_get(int ndevices, struct starpu_perfmodel_device *devices)
{
	int ret;
	STARPU_PTHREAD_RWLOCK_RDLOCK(&arch_combs_mutex);
	ret = _starpu_perfmodel_arch_comb_get(ndevices, devices);
	STARPU_PTHREAD_RWLOCK_UNLOCK(&arch_combs_mutex);
	return ret;
}

int starpu_perfmodel_arch_comb_add(int ndevices, struct starpu_perfmodel_device* devices)
{
	STARPU_PTHREAD_RWLOCK_WRLOCK(&arch_combs_mutex);
	int comb = _starpu_perfmodel_arch_comb_get(ndevices, devices);
	if (comb != -1)
	{
		/* Somebody else added it in between */
		STARPU_PTHREAD_RWLOCK_UNLOCK(&arch_combs_mutex);
		return comb;
	}
	if (current_arch_comb >= nb_arch_combs)
	{
		// We need to allocate more arch_combs
		nb_arch_combs = current_arch_comb+10;
		_STARPU_REALLOC(arch_combs, nb_arch_combs*sizeof(struct starpu_perfmodel_arch*));
	}
	_STARPU_MALLOC(arch_combs[current_arch_comb], sizeof(struct starpu_perfmodel_arch));
	_STARPU_MALLOC(arch_combs[current_arch_comb]->devices, ndevices*sizeof(struct starpu_perfmodel_device));
	arch_combs[current_arch_comb]->ndevices = ndevices;
	int dev;
	for(dev = 0; dev < ndevices; dev++)
	{
		arch_combs[current_arch_comb]->devices[dev].type = devices[dev].type;
		arch_combs[current_arch_comb]->devices[dev].devid = devices[dev].devid;
		arch_combs[current_arch_comb]->devices[dev].ncores = devices[dev].ncores;
	}
	comb = current_arch_comb++;
	STARPU_PTHREAD_RWLOCK_UNLOCK(&arch_combs_mutex);
	return comb;
}

void _starpu_free_arch_combs(void)
{
	int i;
	STARPU_PTHREAD_RWLOCK_WRLOCK(&arch_combs_mutex);
	for(i = 0; i < current_arch_comb; i++)
	{
		free(arch_combs[i]->devices);
		free(arch_combs[i]);
	}
	current_arch_comb = 0;
	free(arch_combs);
	arch_combs = NULL;
	STARPU_PTHREAD_RWLOCK_UNLOCK(&arch_combs_mutex);
	STARPU_PTHREAD_RWLOCK_DESTROY(&arch_combs_mutex);
}

int starpu_perfmodel_get_narch_combs()
{
	return current_arch_comb;
}

struct starpu_perfmodel_arch *starpu_perfmodel_arch_comb_fetch(int comb)
{
	return arch_combs[comb];
}

size_t _starpu_job_get_data_size(struct starpu_perfmodel *model, struct starpu_perfmodel_arch* arch, unsigned impl, struct _starpu_job *j)
{
	struct starpu_task *task = j->task;
	int comb = starpu_perfmodel_arch_comb_get(arch->ndevices, arch->devices);

	if (model && model->state->per_arch && comb != -1 && model->state->per_arch[comb] && model->state->per_arch[comb][impl].size_base)
	{
		return model->state->per_arch[comb][impl].size_base(task, arch, impl);
	}
	else if (model && model->size_base)
	{
		return model->size_base(task, impl);
	}
	else
	{
		unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(task);
		size_t size = 0;

		unsigned buffer;
		for (buffer = 0; buffer < nbuffers; buffer++)
		{
			starpu_data_handle_t handle = STARPU_TASK_GET_HANDLE(task, buffer);
			size += _starpu_data_get_size(handle);
		}
		return size;
	}
}

/*
 * History based model
 */
static void insert_history_entry(struct starpu_perfmodel_history_entry *entry, struct starpu_perfmodel_history_list **list, struct starpu_perfmodel_history_table **history_ptr)
{
	struct starpu_perfmodel_history_list *link;
	struct starpu_perfmodel_history_table *table;

	_STARPU_MALLOC(link, sizeof(struct starpu_perfmodel_history_list));
	link->next = *list;
	link->entry = entry;
	*list = link;

	/* detect concurrency issue */
	//HASH_FIND_UINT32_T(*history_ptr, &entry->footprint, table);
	//STARPU_ASSERT(table == NULL);

	_STARPU_MALLOC(table, sizeof(*table));
	table->footprint = entry->footprint;
	table->history_entry = entry;
	HASH_ADD_UINT32_T(*history_ptr, footprint, table);
}

#ifndef STARPU_SIMGRID
static void check_reg_model(struct starpu_perfmodel *model, int comb, int impl)
{
	struct starpu_perfmodel_per_arch *per_arch_model = &model->state->per_arch[comb][impl];
	struct starpu_perfmodel_regression_model *reg_model = &per_arch_model->regression;

	/*
	 * Linear Regression model
	 */

	/* Unless we have enough measurements, we put NaN in the file to indicate the model is invalid */
	double alpha = nan(""), beta = nan("");
	if (model->type == STARPU_REGRESSION_BASED || model->type == STARPU_NL_REGRESSION_BASED)
	{
		if (reg_model->nsample > 1)
		{
			alpha = reg_model->alpha;
			beta = reg_model->beta;
		}
	}

	/* TODO: check:
	 * reg_model->sumlnx
	 * reg_model->sumlnx2
	 * reg_model->sumlny
	 * reg_model->sumlnxlny
	 * alpha
	 * beta
	 * reg_model->minx
	 * reg_model->maxx
	 */
	(void)alpha;
	(void)beta;

	/*
	 * Non-Linear Regression model
	 */

	double a = nan(""), b = nan(""), c = nan("");

	if (model->type == STARPU_NL_REGRESSION_BASED)
		_starpu_regression_non_linear_power(per_arch_model->list, &a, &b, &c);

	/* TODO: check:
	 * a
	 * b
	 * c
	 */

	/*
	 * Multiple Regression Model
	 */

	if (model->type == STARPU_MULTIPLE_REGRESSION_BASED)
	{
		/* TODO: check: */
	}
}

static void dump_reg_model(FILE *f, struct starpu_perfmodel *model, int comb, int impl)
{
	struct starpu_perfmodel_per_arch *per_arch_model;

	per_arch_model = &model->state->per_arch[comb][impl];
	struct starpu_perfmodel_regression_model *reg_model;
	reg_model = &per_arch_model->regression;

	/*
	 * Linear Regression model
	 */

	/* Unless we have enough measurements, we put NaN in the file to indicate the model is invalid */
	double alpha = nan(""), beta = nan("");
	if (model->type == STARPU_REGRESSION_BASED || model->type == STARPU_NL_REGRESSION_BASED)
	{
		if (reg_model->nsample > 1)
		{
			alpha = reg_model->alpha;
			beta = reg_model->beta;
		}
	}

	fprintf(f, "# sumlnx\tsumlnx2\t\tsumlny\t\tsumlnxlny\talpha\t\tbeta\t\tn\tminx\t\tmaxx\n");
	fprintf(f, "%-15e\t%-15e\t%-15e\t%-15e\t", reg_model->sumlnx, reg_model->sumlnx2, reg_model->sumlny, reg_model->sumlnxlny);
	_starpu_write_double(f, "%-15e", alpha);
	fprintf(f, "\t");
	_starpu_write_double(f, "%-15e", beta);
	fprintf(f, "\t%u\t%-15lu\t%-15lu\n", reg_model->nsample, reg_model->minx, reg_model->maxx);

	/*
	 * Non-Linear Regression model
	 */

	double a = nan(""), b = nan(""), c = nan("");

	if (model->type == STARPU_NL_REGRESSION_BASED)
	{
		if (_starpu_regression_non_linear_power(per_arch_model->list, &a, &b, &c) != 0)
			_STARPU_DISP("Warning: could not compute a non-linear regression for model %s\n", model->symbol);
	}

	fprintf(f, "# a\t\tb\t\tc\n");
	_starpu_write_double(f, "%-15e", a);
	fprintf(f, "\t");
	_starpu_write_double(f, "%-15e", b);
	fprintf(f, "\t");
	_starpu_write_double(f, "%-15e", c);
	fprintf(f, "\n");

	/*
	 * Multiple Regression Model
	 */

	if (model->type != STARPU_MULTIPLE_REGRESSION_BASED)
	{
		fprintf(f, "# not multiple-regression-base\n");
		fprintf(f, "0\n");
	}
	else
	{
		if (reg_model->ncoeff==0 && model->ncombinations!=0 && model->combinations!=NULL)
		{
			reg_model->ncoeff = model->ncombinations + 1;
		}

		_STARPU_MALLOC(reg_model->coeff,  reg_model->ncoeff*sizeof(double));
		_starpu_multiple_regression(per_arch_model->list, reg_model->coeff, reg_model->ncoeff, model->nparameters, model->parameters_names, model->combinations, model->symbol);

		fprintf(f, "# n\tintercept\t");
		if (reg_model->ncoeff==0 || model->ncombinations==0 || model->combinations==NULL)
			fprintf(f, "\n1\tnan");
		else
		{
			unsigned i;
			for (i=0; i < model->ncombinations; i++)
			{
				if (model->parameters_names == NULL)
					fprintf(f, "c%u", i+1);
				else
				{
					unsigned j;
					int first=1;
					for(j=0; j < model->nparameters; j++)
					{
						if (model->combinations[i][j] > 0)
						{
							if (first)
								first=0;
							else
								fprintf(f, "*");

							if(model->parameters_names[j] != NULL)
								fprintf(f, "%s", model->parameters_names[j]);
							else
								fprintf(f, "P%u", j);

							if (model->combinations[i][j] > 1)
								fprintf(f, "^%d", model->combinations[i][j]);
						}
					}
				}
				fprintf(f, "\t\t");
			}

			fprintf(f, "\n%u", reg_model->ncoeff);
			for (i=0; i < reg_model->ncoeff; i++)
				fprintf(f, "\t%-15e", reg_model->coeff[i]);
		}
	}
}
#endif

static void scan_reg_model(FILE *f, const char *path, struct starpu_perfmodel_regression_model *reg_model)
{
	int res;

	/*
	 * Linear Regression model
	 */

	_starpu_drop_comments(f);

	res = fscanf(f, "%le\t%le\t%le\t%le\t", &reg_model->sumlnx, &reg_model->sumlnx2, &reg_model->sumlny, &reg_model->sumlnxlny);
	STARPU_ASSERT_MSG(res == 4, "Incorrect performance model file %s", path);
	res = _starpu_read_double(f, "%le", &reg_model->alpha);
	STARPU_ASSERT_MSG(res == 1, "Incorrect performance model file %s", path);
	res = _starpu_read_double(f, "\t%le", &reg_model->beta);
	STARPU_ASSERT_MSG(res == 1, "Incorrect performance model file %s", path);
	res = fscanf(f, "\t%u\t%lu\t%lu\n", &reg_model->nsample, &reg_model->minx, &reg_model->maxx);
	STARPU_ASSERT_MSG(res == 3, "Incorrect performance model file %s", path);

	/* If any of the parameters describing the linear regression model is NaN, the model is invalid */
	unsigned invalid = (isnan(reg_model->alpha)||isnan(reg_model->beta));
	reg_model->valid = !invalid && VALID_REGRESSION(reg_model);

	/*
	 * Non-Linear Regression model
	 */

	_starpu_drop_comments(f);

	res = _starpu_read_double(f, "%le", &reg_model->a);
	STARPU_ASSERT_MSG(res == 1, "Incorrect performance model file %s", path);
	res = _starpu_read_double(f, "\t%le", &reg_model->b);
	STARPU_ASSERT_MSG(res == 1, "Incorrect performance model file %s", path);
	res = _starpu_read_double(f, "%le", &reg_model->c);
	STARPU_ASSERT_MSG(res == 1, "Incorrect performance model file %s", path);
	res = fscanf(f, "\n");
	STARPU_ASSERT_MSG(res == 0, "Incorrect performance model file %s", path);

	/* If any of the parameters describing the non-linear regression model is NaN, the model is invalid */
	unsigned nl_invalid = (isnan(reg_model->a)||isnan(reg_model->b)||isnan(reg_model->c));
	reg_model->nl_valid = !nl_invalid && VALID_REGRESSION(reg_model);

	_starpu_drop_comments(f);

	// Read how many coefficients is there
	res = fscanf(f, "%u", &reg_model->ncoeff);
	STARPU_ASSERT_MSG(res == 1, "Incorrect performance model file %s", path);

	/*
	 * Multiple Regression Model
	 */
	if (reg_model->ncoeff != 0)
	{
		_STARPU_MALLOC(reg_model->coeff, reg_model->ncoeff*sizeof(double));

		unsigned multi_invalid = 0;
		unsigned i;
		for (i=0; i < reg_model->ncoeff; i++)
		{
			res = _starpu_read_double(f, "%le", &reg_model->coeff[i]);
			STARPU_ASSERT_MSG(res == 1, "Incorrect performance model file %s", path);
			multi_invalid = (multi_invalid||isnan(reg_model->coeff[i]));
		}
		reg_model->multi_valid = !multi_invalid;
	}
	res = fscanf(f, "\n");
	STARPU_ASSERT_MSG(res == 0, "Incorrect performance model file %s", path);
}


#ifndef STARPU_SIMGRID
static void check_history_entry(struct starpu_perfmodel_history_entry *entry)
{
	STARPU_ASSERT_MSG(entry->deviation >= 0, "entry=%p, entry->deviation=%lf\n", entry, entry->deviation);
	STARPU_ASSERT_MSG(entry->sum >= 0, "entry=%p, entry->sum=%lf\n", entry, entry->sum);
	STARPU_ASSERT_MSG(entry->sum2 >= 0, "entry=%p, entry->sum2=%lf\n", entry, entry->sum2);
	STARPU_ASSERT_MSG(entry->mean >= 0, "entry=%p, entry->mean=%lf\n", entry, entry->mean);
	STARPU_ASSERT_MSG(isnan(entry->flops)||entry->flops >= 0, "entry=%p, entry->flops=%lf\n", entry, entry->flops);
	STARPU_ASSERT_MSG(entry->duration >= 0, "entry=%p, entry->duration=%lf\n", entry, entry->duration);
}
static void dump_history_entry(FILE *f, struct starpu_perfmodel_history_entry *entry)
{
	fprintf(f, "%08x\t%-15lu\t%-15e\t%-15e\t%-15e\t%-15e\t%-15e\t%u\n", entry->footprint, (unsigned long) entry->size, entry->flops, entry->mean, entry->deviation, entry->sum, entry->sum2, entry->nsample);
}
#endif

static void scan_history_entry(FILE *f, const char *path, struct starpu_perfmodel_history_entry *entry)
{
	int res;

	_starpu_drop_comments(f);

	/* In case entry is NULL, we just drop these values */
	unsigned nsample;
	uint32_t footprint;
	unsigned long size; /* in bytes */
	double flops;
	double mean;
	double deviation;
	double sum;
	double sum2;

	char line[STR_LONG_LENGTH];
	char *ret;

	ret = fgets(line, sizeof(line), f);
	STARPU_ASSERT(ret);
	STARPU_ASSERT(strchr(line, '\n'));

	/* Read the values from the file */
	res = sscanf(line, "%x\t%lu\t%le\t%le\t%le\t%le\t%le\t%u", &footprint, &size, &flops, &mean, &deviation, &sum, &sum2, &nsample);

	if (res != 8)
	{
		flops = 0.;
		/* Read the values from the file */
		res = sscanf(line, "%x\t%lu\t%le\t%le\t%le\t%le\t%u", &footprint, &size, &mean, &deviation, &sum, &sum2, &nsample);
		STARPU_ASSERT_MSG(res == 7, "Incorrect performance model file %s", path);
	}

	if (entry)
	{
		STARPU_ASSERT_MSG(isnan(flops) || flops >=0, "Negative flops %lf in performance model file %s", flops, path);
		STARPU_ASSERT_MSG(mean >=0, "Negative mean %lf in performance model file %s", mean, path);
		STARPU_ASSERT_MSG(deviation >=0, "Negative deviation %lf in performance model file %s", deviation, path);
		STARPU_ASSERT_MSG(sum >=0, "Negative sum %lf in performance model file %s", sum, path);
		STARPU_ASSERT_MSG(sum2 >=0, "Negative sum2 %lf in performance model file %s", sum2, path);
		entry->footprint = footprint;
		entry->size = size;
		entry->flops = flops;
		entry->mean = mean;
		entry->deviation = deviation;
		entry->sum = sum;
		entry->sum2 = sum2;
		entry->nsample = nsample;
	}
}

static void parse_per_arch_model_file(FILE *f, const char *path, struct starpu_perfmodel_per_arch *per_arch_model, unsigned scan_history, struct starpu_perfmodel *model)
{
	unsigned nentries;
	struct starpu_perfmodel_regression_model *reg_model = &per_arch_model->regression;

	_starpu_drop_comments(f);

	int res = fscanf(f, "%u\n", &nentries);
	STARPU_ASSERT_MSG(res == 1, "Incorrect performance model file %s", path);

	scan_reg_model(f, path, reg_model);

	/* parse entries */
	unsigned i;
	for (i = 0; i < nentries; i++)
	{
		struct starpu_perfmodel_history_entry *entry = NULL;
		if (scan_history)
		{
			_STARPU_CALLOC(entry, 1, sizeof(struct starpu_perfmodel_history_entry));

			/* Tell  helgrind that we do not care about
			 * racing access to the sampling, we only want a
			 * good-enough estimation */
			STARPU_HG_DISABLE_CHECKING(entry->nsample);
			STARPU_HG_DISABLE_CHECKING(entry->mean);
			entry->nerror = 0;
		}

		scan_history_entry(f, path, entry);

		/* insert the entry in the hashtable and the list structures  */
		/* TODO: Insert it at the end of the list, to avoid reversing
		 * the order... But efficiently! We may have a lot of entries */
		if (scan_history)
			insert_history_entry(entry, &per_arch_model->list, &per_arch_model->history);
	}

	if (model && model->type == STARPU_PERFMODEL_INVALID)
	{
		/* Tool loading a perfmodel without having the corresponding codelet */
		if (reg_model->ncoeff != 0)
			model->type = STARPU_MULTIPLE_REGRESSION_BASED;
		else if (!isnan(reg_model->a) && !isnan(reg_model->b) && !isnan(reg_model->c))
			model->type = STARPU_NL_REGRESSION_BASED;
		else if (!isnan(reg_model->alpha) && !isnan(reg_model->beta))
			model->type = STARPU_REGRESSION_BASED;
		else if (nentries)
			model->type = STARPU_HISTORY_BASED;
		/* else unknown, leave invalid */
	}
}


static void parse_arch(FILE *f, const char *path, struct starpu_perfmodel *model, unsigned scan_history, int comb)
{
	struct starpu_perfmodel_per_arch dummy;
	unsigned nimpls, impl, i, ret;

	/* Parsing number of implementation */
	_starpu_drop_comments(f);
	ret = fscanf(f, "%u\n", &nimpls);
	STARPU_ASSERT_MSG(ret == 1, "Incorrect performance model file %s", path);

	if( model != NULL)
	{
		/* Parsing each implementation */
		unsigned implmax = STARPU_MIN(nimpls, STARPU_MAXIMPLEMENTATIONS);
		model->state->nimpls[comb] = implmax;
		if (!model->state->per_arch[comb])
		{
			_starpu_perfmodel_malloc_per_arch(model, comb, STARPU_MAXIMPLEMENTATIONS);
		}
		if (!model->state->per_arch_is_set[comb])
		{
			_starpu_perfmodel_malloc_per_arch_is_set(model, comb, STARPU_MAXIMPLEMENTATIONS);
		}

		for (impl = 0; impl < implmax; impl++)
		{
			struct starpu_perfmodel_per_arch *per_arch_model = &model->state->per_arch[comb][impl];
			model->state->per_arch_is_set[comb][impl] = 1;
			parse_per_arch_model_file(f, path, per_arch_model, scan_history, model);
		}
	}
	else
	{
		impl = 0;
	}

	/* if the number of implementation is greater than STARPU_MAXIMPLEMENTATIONS
	 * we skip the last implementation */
	for (i = impl; i < nimpls; i++)
		parse_per_arch_model_file(f, path, &dummy, 0, NULL);
}

static void parse_comb(FILE *f, const char *path, struct starpu_perfmodel *model, unsigned scan_history, int comb)
{
	int ndevices = 0;
	_starpu_drop_comments(f);
	int ret = fscanf(f, "%d\n", &ndevices );
	STARPU_ASSERT_MSG(ret == 1, "Incorrect performance model file %s", path);

	struct starpu_perfmodel_device devices[ndevices];

	int dev;
	for(dev = 0; dev < ndevices; dev++)
	{
		_starpu_drop_comments(f);
		int type;
		ret = fscanf(f, "%d\n", &type);
		STARPU_ASSERT_MSG(ret == 1, "Incorrect performance model file %s", path);
		int dev_id;
		_starpu_drop_comments(f);
		ret = fscanf(f, "%d\n", &dev_id);
		STARPU_ASSERT_MSG(ret == 1, "Incorrect performance model file %s", path);
		int ncores;
		_starpu_drop_comments(f);
		ret = fscanf(f, "%d\n", &ncores);
		STARPU_ASSERT_MSG(ret == 1, "Incorrect performance model file %s", path);
		devices[dev].type = type;
		devices[dev].devid = dev_id;
		devices[dev].ncores = ncores;
	}
	int id_comb = starpu_perfmodel_arch_comb_get(ndevices, devices);
	if(id_comb == -1)
		id_comb = starpu_perfmodel_arch_comb_add(ndevices, devices);

	model->state->combs[comb] = id_comb;
	parse_arch(f, path, model, scan_history, id_comb);
}

static int parse_model_file(FILE *f, const char *path, struct starpu_perfmodel *model, unsigned scan_history)
{
	int ret, version=0;

        /* First check that it's not empty (very common corruption result, for
         * which there is no solution) */
	fseek(f, 0, SEEK_END);
	long pos = ftell(f);
	if (pos == 0)
	{
		_STARPU_DISP("Performance model file %s is empty, ignoring it\n", path);
		return 1;
	}
	rewind(f);

	/* Parsing performance model version */
	_starpu_drop_comments(f);
	ret = fscanf(f, "%d\n", &version);
	STARPU_ASSERT_MSG(version == _STARPU_PERFMODEL_VERSION, "Incorrect performance model file %s with a model version %d not being the current model version (%d)\n", path,
			  version, _STARPU_PERFMODEL_VERSION);
	STARPU_ASSERT_MSG(ret == 1, "Incorrect performance model file %s", path);

	int ncombs = 0;
	_starpu_drop_comments(f);
	ret = fscanf(f, "%d\n", &ncombs);
	STARPU_ASSERT_MSG(ret == 1, "Incorrect performance model file %s", path);
	if(ncombs > 0)
	{
		model->state->ncombs = ncombs;
	}

	if (ncombs > model->state->ncombs_set)
	{
		// The model has more combs than the original number of arch_combs, we need to reallocate
		_starpu_perfmodel_realloc(model, ncombs);
	}

	int comb;
	for(comb = 0; comb < ncombs; comb++)
		parse_comb(f, path, model, scan_history, comb);

	return 0;
}

#ifndef STARPU_SIMGRID
static void check_per_arch_model(struct starpu_perfmodel *model, int comb, unsigned impl)
{
	struct starpu_perfmodel_per_arch *per_arch_model;

	per_arch_model = &model->state->per_arch[comb][impl];
	/* count the number of elements in the lists */
	struct starpu_perfmodel_history_list *ptr = NULL;
	unsigned nentries = 0;

	if (model->type == STARPU_HISTORY_BASED || model->type == STARPU_NL_REGRESSION_BASED  || model->type == STARPU_REGRESSION_BASED)
	{
		/* Dump the list of all entries in the history */
		ptr = per_arch_model->list;
		while(ptr)
		{
			nentries++;
			ptr = ptr->next;
		}
	}

	/* header */
	char archname[STR_SHORT_LENGTH];
	starpu_perfmodel_get_arch_name(arch_combs[comb], archname,  sizeof(archname), impl);
	STARPU_ASSERT(strlen(archname)>0);
	check_reg_model(model, comb, impl);

	/* Dump the history into the model file in case it is necessary */
	if (model->type == STARPU_HISTORY_BASED || model->type == STARPU_NL_REGRESSION_BASED || model->type == STARPU_REGRESSION_BASED)
	{
		ptr = per_arch_model->list;
		while (ptr)
		{
			check_history_entry(ptr->entry);
			ptr = ptr->next;
		}
	}
}
static void dump_per_arch_model_file(FILE *f, struct starpu_perfmodel *model, int comb, unsigned impl)
{
	struct starpu_perfmodel_per_arch *per_arch_model;

	per_arch_model = &model->state->per_arch[comb][impl];
	/* count the number of elements in the lists */
	struct starpu_perfmodel_history_list *ptr = NULL;
	unsigned nentries = 0;

       if (model->type == STARPU_HISTORY_BASED || model->type == STARPU_NL_REGRESSION_BASED || model->type == STARPU_REGRESSION_BASED)
	{
		/* Dump the list of all entries in the history */
		ptr = per_arch_model->list;
		while(ptr)
		{
			nentries++;
			ptr = ptr->next;
		}
	}

	/* header */
	char archname[STR_SHORT_LENGTH];
	starpu_perfmodel_get_arch_name(arch_combs[comb], archname,  sizeof(archname), impl);
	fprintf(f, "#####\n");
	fprintf(f, "# Model for %s\n", archname);
	fprintf(f, "# number of entries\n%u\n", nentries);

	dump_reg_model(f, model, comb, impl);

	/* Dump the history into the model file in case it is necessary */
       if (model->type == STARPU_HISTORY_BASED || model->type == STARPU_NL_REGRESSION_BASED || model->type == STARPU_REGRESSION_BASED)
	{
		fprintf(f, "# hash\t\tsize\t\tflops\t\tmean (us)\tdev (us)\tsum\t\tsum2\t\tn\n");
		ptr = per_arch_model->list;
		while (ptr)
		{
			dump_history_entry(f, ptr->entry);
			ptr = ptr->next;
		}
	}

	fprintf(f, "\n");
}

static void check_model(struct starpu_perfmodel *model)
{
	int ncombs = model->state->ncombs;
	STARPU_ASSERT(ncombs >= 0);

	int i, impl, dev;
	for(i = 0; i < ncombs; i++)
	{
		int comb = model->state->combs[i];
		STARPU_ASSERT(comb >= 0);

		int ndevices = arch_combs[comb]->ndevices;
		STARPU_ASSERT(ndevices >= 1);

		for(dev = 0; dev < ndevices; dev++)
		{
			STARPU_ASSERT(arch_combs[comb]->devices[dev].type >= 0);
			STARPU_ASSERT(arch_combs[comb]->devices[dev].type <= 5);

			STARPU_ASSERT(arch_combs[comb]->devices[dev].devid >= 0);

			STARPU_ASSERT(arch_combs[comb]->devices[dev].ncores >= 0);
		}

		int nimpls = model->state->nimpls[comb];
		STARPU_ASSERT(nimpls >= 1);
		for (impl = 0; impl < nimpls; impl++)
		{
			check_per_arch_model(model, comb, impl);
		}
	}
}

static void dump_model_file(FILE *f, struct starpu_perfmodel *model)
{
	fprintf(f, "##################\n");
	fprintf(f, "# Performance Model Version\n");
	fprintf(f, "%d\n\n", _STARPU_PERFMODEL_VERSION);

	int ncombs = model->state->ncombs;

	fprintf(f, "####################\n");
	fprintf(f, "# COMBs\n");
	fprintf(f, "# number of combinations\n");
	fprintf(f, "%d\n", ncombs);

	int i, impl, dev;
	for(i = 0; i < ncombs; i++)
	{
		int comb = model->state->combs[i];
		int ndevices = arch_combs[comb]->ndevices;
		fprintf(f, "####################\n");
		fprintf(f, "# COMB_%d\n", comb);
		fprintf(f, "# number of types devices\n");
		fprintf(f, "%d\n", ndevices);

		for(dev = 0; dev < ndevices; dev++)
		{
			fprintf(f, "####################\n");
			fprintf(f, "# DEV_%d\n", dev);
			fprintf(f, "# device type (CPU - %d, CUDA - %d, OPENCL - %d, MIC - %d, MPI_MS - %d)\n",
				STARPU_CPU_WORKER, STARPU_CUDA_WORKER, STARPU_OPENCL_WORKER, STARPU_MIC_WORKER, STARPU_MPI_MS_WORKER);
			fprintf(f, "%u\n", arch_combs[comb]->devices[dev].type);

			fprintf(f, "####################\n");
			fprintf(f, "# DEV_%d\n", dev);
			fprintf(f, "# device id \n");
			fprintf(f, "%u\n", arch_combs[comb]->devices[dev].devid);

			fprintf(f, "####################\n");
			fprintf(f, "# DEV_%d\n", dev);
			fprintf(f, "# number of cores \n");
			fprintf(f, "%u\n", arch_combs[comb]->devices[dev].ncores);
		}

		int nimpls = model->state->nimpls[comb];
		fprintf(f, "##########\n");
		fprintf(f, "# number of implementations\n");
		fprintf(f, "%d\n", nimpls);
		for (impl = 0; impl < nimpls; impl++)
		{
			dump_per_arch_model_file(f, model, comb, impl);
		}
	}
}
#endif

static void dump_history_entry_xml(FILE *f, struct starpu_perfmodel_history_entry *entry)
{
	fprintf(f, "      <entry footprint=\"%08x\" size=\"%lu\" flops=\"%e\" mean=\"%e\" deviation=\"%e\" sum=\"%e\" sum2=\"%e\" nsample=\"%u\"/>\n", entry->footprint, (unsigned long) entry->size, entry->flops, entry->mean, entry->deviation, entry->sum, entry->sum2, entry->nsample);
}

static void dump_reg_model_xml(FILE *f, struct starpu_perfmodel *model, int comb, int impl)
{
	struct starpu_perfmodel_per_arch *per_arch_model;

	per_arch_model = &model->state->per_arch[comb][impl];
	struct starpu_perfmodel_regression_model *reg_model = &per_arch_model->regression;

	/*
	 * Linear Regression model
	 */

	if (model->type == STARPU_REGRESSION_BASED)
	{
		fprintf(f, "      <!-- time = alpha size ^ beta -->\n");
		fprintf(f, "      <l_regression sumlnx=\"%e\" sumlnx2=\"%e\" sumlny=\"%e\" sumlnxlny=\"%e\"", reg_model->sumlnx, reg_model->sumlnx2, reg_model->sumlny, reg_model->sumlnxlny);
		fprintf(f, " alpha=\"");
		_starpu_write_double(f, "%e", reg_model->alpha);
		fprintf(f, "\" beta=\"");
		_starpu_write_double(f, "%e", reg_model->beta);
		fprintf(f, "\" nsample=\"%u\" minx=\"%lu\" maxx=\"%lu\"/>\n", reg_model->nsample, reg_model->minx, reg_model->maxx);
	}

	/*
	 * Non-Linear Regression model
	 */

	else if (model->type == STARPU_NL_REGRESSION_BASED)
	{
		fprintf(f, "      <!-- time = a size ^b + c -->\n");
		fprintf(f, "      <nl_regression a=\"");
		_starpu_write_double(f, "%e", reg_model->a);
		fprintf(f, "\" b=\"");
		_starpu_write_double(f, "%e", reg_model->b);
		fprintf(f, "\" c=\"");
		_starpu_write_double(f, "%e", reg_model->c);
		fprintf(f, "\"/>\n");
	}

	else if (model->type == STARPU_MULTIPLE_REGRESSION_BASED)
	{
		if (reg_model->ncoeff==0 || model->ncombinations==0 || model->combinations==NULL)
			fprintf(f, "      <ml_regression constant=\"nan\"/>\n");
		else
		{
			unsigned i;
			fprintf(f, "      <ml_regression constant=\"%e\">\n", reg_model->coeff[0]);
			for (i=0; i < model->ncombinations; i++)
			{
				fprintf(f, "        <monomial name=\"");
				if (model->parameters_names == NULL)
					fprintf(f, "c%u", i+1);
				else
				{
					unsigned j;
					int first=1;
					for(j=0; j < model->nparameters; j++)
					{
						if (model->combinations[i][j] > 0)
						{
							if (first)
								first=0;
							else
								fprintf(f, "*");

							if(model->parameters_names[j] != NULL)
								fprintf(f, "%s", model->parameters_names[j]);
							else
								fprintf(f, "P%u", j);

							if (model->combinations[i][j] > 1)
								fprintf(f, "^%d", model->combinations[i][j]);
						}
					}
				}
				fprintf(f, "\" coef=\"%e\"/>\n", reg_model->coeff[i+1]);
			}
			fprintf(f, "      </ml_regression>\n");
		}
	}
}

static void dump_per_arch_model_xml(FILE *f, struct starpu_perfmodel *model, int comb, unsigned impl)
{
	struct starpu_perfmodel_per_arch *per_arch_model;

	per_arch_model = &model->state->per_arch[comb][impl];
	/* count the number of elements in the lists */
	struct starpu_perfmodel_history_list *ptr;

	dump_reg_model_xml(f, model, comb, impl);

	/* Dump the history into the model file in case it is necessary */
	ptr = per_arch_model->list;
	while (ptr)
	{
		dump_history_entry_xml(f, ptr->entry);
		ptr = ptr->next;
	}
}

void starpu_perfmodel_dump_xml(FILE *f, struct starpu_perfmodel *model)
{
	_starpu_init_and_load_perfmodel(model);

	fprintf(f, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
	fprintf(f, "<!DOCTYPE StarPUPerfmodel SYSTEM \"starpu-perfmodel.dtd\">\n");
	fprintf(f, "<!-- symbol %s -->\n", model->symbol);
	fprintf(f, "<!-- All times in us -->\n");
	fprintf(f, "<perfmodel version=\"%u\">\n", _STARPU_PERFMODEL_VERSION);

	STARPU_PTHREAD_RWLOCK_RDLOCK(&model->state->model_rwlock);
	int ncombs = model->state->ncombs;
	int i, impl, dev;

	for(i = 0; i < ncombs; i++)
	{
		int comb = model->state->combs[i];
		int ndevices = arch_combs[comb]->ndevices;

		fprintf(f, "  <combination>\n");
		for(dev = 0; dev < ndevices; dev++)
		{
			const char *type;
			switch (arch_combs[comb]->devices[dev].type)
			{
				case STARPU_CPU_WORKER: type = "CPU"; break;
				case STARPU_CUDA_WORKER: type = "CUDA"; break;
				case STARPU_OPENCL_WORKER: type = "OpenCL"; break;
				case STARPU_MIC_WORKER: type = "MIC"; break;
				case STARPU_MPI_MS_WORKER: type = "MPI_MS"; break;
				default: STARPU_ASSERT(0);
			}
			fprintf(f, "    <device type=\"%s\" id=\"%d\"",
					type,
					arch_combs[comb]->devices[dev].devid);
			if (arch_combs[comb]->devices[dev].type == STARPU_CPU_WORKER)
				fprintf(f, " ncores=\"%d\"",
						arch_combs[comb]->devices[dev].ncores);
			fprintf(f, "/>\n");
		}
		int nimpls = model->state->nimpls[comb];
		for (impl = 0; impl < nimpls; impl++)
		{
			fprintf(f, "    <implementation id=\"%d\">\n", impl);
			char archname[STR_SHORT_LENGTH];
			starpu_perfmodel_get_arch_name(arch_combs[comb], archname,  sizeof(archname), impl);
			fprintf(f, "      <!-- %s -->\n", archname);
			dump_per_arch_model_xml(f, model, comb, impl);
			fprintf(f, "    </implementation>\n");
		}
		fprintf(f, "  </combination>\n");
	}
	STARPU_PTHREAD_RWLOCK_UNLOCK(&model->state->model_rwlock);
	fprintf(f, "</perfmodel>\n");
}

void _starpu_perfmodel_realloc(struct starpu_perfmodel *model, int nb)
{
	int i;

	STARPU_ASSERT(nb > model->state->ncombs_set);
#ifdef SSIZE_MAX
	STARPU_ASSERT((size_t) nb < SSIZE_MAX / sizeof(struct starpu_perfmodel_per_arch*));
#endif
	_STARPU_REALLOC(model->state->per_arch, nb*sizeof(struct starpu_perfmodel_per_arch*));
	_STARPU_REALLOC(model->state->per_arch_is_set, nb*sizeof(int*));
	_STARPU_REALLOC(model->state->nimpls, nb*sizeof(int));
	_STARPU_REALLOC(model->state->nimpls_set, nb*sizeof(int));
	_STARPU_REALLOC(model->state->combs, nb*sizeof(int));
	for(i = model->state->ncombs_set; i < nb; i++)
	{
		model->state->per_arch[i] = NULL;
		model->state->per_arch_is_set[i] = NULL;
		model->state->nimpls[i] = 0;
		model->state->nimpls_set[i] = 0;
	}
	model->state->ncombs_set = nb;
}

void starpu_perfmodel_init(struct starpu_perfmodel *model)
{
	int already_init;
	int ncombs;

	STARPU_ASSERT(model);

	STARPU_PTHREAD_RWLOCK_RDLOCK(&registered_models_rwlock);
	already_init = model->is_init;
	STARPU_PTHREAD_RWLOCK_UNLOCK(&registered_models_rwlock);

	if (already_init)
		return;

	/* The model is still not loaded so we grab the lock in write mode, and
	 * if it's not loaded once we have the lock, we do load it. */
	STARPU_PTHREAD_RWLOCK_WRLOCK(&registered_models_rwlock);

	/* Was the model initialized since the previous test ? */
	if (model->is_init)
	{
		STARPU_PTHREAD_RWLOCK_UNLOCK(&registered_models_rwlock);
		return;
	}

	_STARPU_MALLOC(model->state, sizeof(struct _starpu_perfmodel_state));
	STARPU_PTHREAD_RWLOCK_INIT(&model->state->model_rwlock, NULL);

	STARPU_PTHREAD_RWLOCK_RDLOCK(&arch_combs_mutex);
	model->state->ncombs_set = ncombs = nb_arch_combs;
	STARPU_PTHREAD_RWLOCK_UNLOCK(&arch_combs_mutex);
	_STARPU_CALLOC(model->state->per_arch, ncombs, sizeof(struct starpu_perfmodel_per_arch*));
	_STARPU_CALLOC(model->state->per_arch_is_set, ncombs, sizeof(int*));
	_STARPU_CALLOC(model->state->nimpls, ncombs, sizeof(int));
	_STARPU_CALLOC(model->state->nimpls_set, ncombs, sizeof(int));
	_STARPU_MALLOC(model->state->combs, ncombs*sizeof(int));
	model->state->ncombs = 0;

	/* add the model to a linked list */
	struct _starpu_perfmodel *node = _starpu_perfmodel_new();

	node->model = model;
	//model->debug_modelid = debug_modelid++;

	/* put this model at the beginning of the list */
	_starpu_perfmodel_list_push_front(&registered_models, node);

	model->is_init = 1;
	STARPU_PTHREAD_RWLOCK_UNLOCK(&registered_models_rwlock);
}

static void get_model_debug_path(struct starpu_perfmodel *model, const char *arch, char *path, size_t maxlen)
{
	STARPU_ASSERT(path);

	char hostname[STR_LONG_LENGTH];
	_starpu_gethostname(hostname, sizeof(hostname));

	snprintf(path, maxlen, "%s/%s.%s.%s.debug", _starpu_get_perf_model_dir_debug(), model->symbol, hostname, arch);
}

void starpu_perfmodel_get_model_path(const char *symbol, char *path, size_t maxlen)
{
	char hostname[STR_LONG_LENGTH];
	_starpu_gethostname(hostname, sizeof(hostname));
	const char *dot = strrchr(symbol, '.');

	snprintf(path, maxlen, "%s/%s%s%s", _starpu_get_perf_model_dir_codelet(), symbol, dot?"":".", dot?"":hostname);
}

#ifndef STARPU_SIMGRID
static void save_history_based_model(struct starpu_perfmodel *model)
{
	STARPU_ASSERT(model);
	STARPU_ASSERT(model->symbol);
	int locked;

	/* TODO checks */

	/* filename = $STARPU_PERF_MODEL_DIR/codelets/symbol.hostname */
	char path[STR_LONG_LENGTH];
	starpu_perfmodel_get_model_path(model->symbol, path, sizeof(path));

	_STARPU_DEBUG("Opening performance model file %s for model %s\n", path, model->symbol);

	/* overwrite existing file, or create it */
	FILE *f;
	f = fopen(path, "w+");
	STARPU_ASSERT_MSG(f, "Could not save performance model %s\n", path);

	locked = _starpu_fwrlock(f) == 0;
	check_model(model);
	_starpu_fftruncate(f, 0);
	dump_model_file(f, model);
	if (locked)
		_starpu_fwrunlock(f);

	fclose(f);
}
#endif

static void _starpu_dump_registered_models(void)
{
#ifndef STARPU_SIMGRID
	STARPU_PTHREAD_RWLOCK_WRLOCK(&registered_models_rwlock);

	struct _starpu_perfmodel *node;

	_STARPU_DEBUG("DUMP MODELS !\n");

	for (node  = _starpu_perfmodel_list_begin(&registered_models);
	     node != _starpu_perfmodel_list_end(&registered_models);
	     node  = _starpu_perfmodel_list_next(node))
	{
		if (node->model->is_init)
			save_history_based_model(node->model);
	}

	STARPU_PTHREAD_RWLOCK_UNLOCK(&registered_models_rwlock);
#endif
}

void starpu_perfmodel_initialize(void)
{
	/* make sure the performance model directory exists (or create it) */
	_starpu_create_sampling_directory_if_needed();

	_starpu_perfmodel_list_init(&registered_models);

	STARPU_PTHREAD_RWLOCK_INIT(&registered_models_rwlock, NULL);
	STARPU_PTHREAD_RWLOCK_INIT(&arch_combs_mutex, NULL);
}

void _starpu_initialize_registered_performance_models(void)
{
	starpu_perfmodel_initialize();

	struct _starpu_machine_config *conf = _starpu_get_machine_config();
	unsigned ncores = conf->topology.nhwcpus;
	unsigned ncuda =  conf->topology.nhwcudagpus;
	unsigned nopencl = conf->topology.nhwopenclgpus;
	unsigned nmic = 0;
	unsigned i;
	for(i = 0; i < conf->topology.nhwmicdevices; i++)
		nmic += conf->topology.nhwmiccores[i];
	unsigned nmpi = 0;
	for(i = 0; i < conf->topology.nhwmpidevices; i++)
		nmpi += conf->topology.nhwmpicores[i];

	// We used to allocate 2**(ncores + ncuda + nopencl + nmic + nmpi), this is too big
	// We now allocate only 2*(ncores + ncuda + nopencl + nmic + nmpi), and reallocate when necessary in starpu_perfmodel_arch_comb_add
	nb_arch_combs = 2 * (ncores + ncuda + nopencl + nmic + nmpi);
	_STARPU_MALLOC(arch_combs, nb_arch_combs*sizeof(struct starpu_perfmodel_arch*));
	current_arch_comb = 0;
	historymaxerror = starpu_get_env_number_default("STARPU_HISTORY_MAX_ERROR", STARPU_HISTORYMAXERROR);
	_starpu_calibration_minimum = starpu_get_env_number_default("STARPU_CALIBRATE_MINIMUM", 10);
	ignore_devid[STARPU_CPU_WORKER] = starpu_get_env_number_default("STARPU_PERF_MODEL_HOMOGENEOUS_CPU", 1);
	ignore_devid[STARPU_CUDA_WORKER] = starpu_get_env_number_default("STARPU_PERF_MODEL_HOMOGENEOUS_CUDA", 0);
	ignore_devid[STARPU_OPENCL_WORKER] = starpu_get_env_number_default("STARPU_PERF_MODEL_HOMOGENEOUS_OPENCL", 0);
	ignore_devid[STARPU_MIC_WORKER] = starpu_get_env_number_default("STARPU_PERF_MODEL_HOMOGENEOUS_MIC", 0);
	ignore_devid[STARPU_MPI_MS_WORKER] = starpu_get_env_number_default("STARPU_PERF_MODEL_HOMOGENEOUS_MPI_MS", 0);
}

void _starpu_deinitialize_performance_model(struct starpu_perfmodel *model)
{
	if(model->is_init && model->state && model->state->per_arch != NULL)
	{
		int i;
		for(i=0 ; i<model->state->ncombs_set ; i++)
		{
			if (model->state->per_arch[i])
			{
				int impl;
				for(impl=0 ; impl<model->state->nimpls_set[i] ; impl++)
				{
					struct starpu_perfmodel_per_arch *archmodel = &model->state->per_arch[i][impl];
					if (archmodel->history)
					{
						struct starpu_perfmodel_history_list *list;
						struct starpu_perfmodel_history_table *entry=NULL, *tmp=NULL;

						HASH_ITER(hh, archmodel->history, entry, tmp)
						{
							HASH_DEL(archmodel->history, entry);
							free(entry);
						}
						archmodel->history = NULL;

						list = archmodel->list;
						while (list)
						{
							struct starpu_perfmodel_history_list *plist;
							free(list->entry);
							plist = list;
							list = list->next;
							free(plist);
						}
						archmodel->list = NULL;
					}
				}
				free(model->state->per_arch[i]);
				model->state->per_arch[i] = NULL;

				free(model->state->per_arch_is_set[i]);
				model->state->per_arch_is_set[i] = NULL;
			}
		}
		free(model->state->per_arch);
		model->state->per_arch = NULL;

		free(model->state->per_arch_is_set);
		model->state->per_arch_is_set = NULL;

		free(model->state->nimpls);
		model->state->nimpls = NULL;

		free(model->state->nimpls_set);
		model->state->nimpls_set = NULL;

		free(model->state->combs);
		model->state->combs = NULL;
		model->state->ncombs = 0;
	}
	model->is_init = 0;
	model->is_loaded = 0;
}

void _starpu_deinitialize_registered_performance_models(void)
{
	if (_starpu_get_calibrate_flag())
		_starpu_dump_registered_models();

	STARPU_PTHREAD_RWLOCK_WRLOCK(&registered_models_rwlock);

	struct _starpu_perfmodel *node, *nnode;

	_STARPU_DEBUG("FREE MODELS !\n");

	for (node  = _starpu_perfmodel_list_begin(&registered_models);
	     node != _starpu_perfmodel_list_end(&registered_models);
	     node  = nnode)
	{
		struct starpu_perfmodel *model = node->model;
		nnode = _starpu_perfmodel_list_next(node);

		STARPU_PTHREAD_RWLOCK_WRLOCK(&model->state->model_rwlock);
		_starpu_deinitialize_performance_model(model);
		STARPU_PTHREAD_RWLOCK_UNLOCK(&model->state->model_rwlock);

		free(node->model->state);
		node->model->state = NULL;

		_starpu_perfmodel_list_erase(&registered_models, node);
		_starpu_perfmodel_delete(node);
	}

	STARPU_PTHREAD_RWLOCK_UNLOCK(&registered_models_rwlock);
	STARPU_PTHREAD_RWLOCK_DESTROY(&registered_models_rwlock);
	starpu_perfmodel_free_sampling();
}

/* We first try to grab the global lock in read mode to check whether the model
 * was loaded or not (this is very likely to have been already loaded). If the
 * model was not loaded yet, we take the lock in write mode, and if the model
 * is still not loaded once we have the lock, we do load it.  */
void _starpu_load_history_based_model(struct starpu_perfmodel *model, unsigned scan_history)
{
	STARPU_PTHREAD_RWLOCK_WRLOCK(&model->state->model_rwlock);

	if(!model->is_loaded)
	{
		char path[STR_LONG_LENGTH];
		// Check if a symbol is defined before trying to load the model from a file
		STARPU_ASSERT_MSG(model->symbol, "history-based performance models must have a symbol");

		starpu_perfmodel_get_model_path(model->symbol, path, sizeof(path));

		_STARPU_DEBUG("Opening performance model file %s for model %s ...\n", path, model->symbol);

		unsigned calibrate_flag = _starpu_get_calibrate_flag();
		model->benchmarking = calibrate_flag;
		model->is_loaded = 1;

		if (calibrate_flag == 2)
		{
			/* The user specified that the performance model should
			 * be overwritten, so we don't load the existing file !
			 * */
			_STARPU_DEBUG("Overwrite existing file\n");
		}
		else
		{
			/* We try to load the file */
			FILE *f;
			f = fopen(path, "r");
			if (f)
			{
				int locked;
				locked = _starpu_frdlock(f) == 0;
				parse_model_file(f, path, model, scan_history);
				if (locked)
					_starpu_frdunlock(f);
				fclose(f);
				_STARPU_DEBUG("Performance model file %s for model %s is loaded\n", path, model->symbol);
			}
			else
			{
				_STARPU_DEBUG("Performance model file %s does not exist or is not readable: %s\n", path, strerror(errno));
			}
		}

	}
	STARPU_PTHREAD_RWLOCK_UNLOCK(&model->state->model_rwlock);

}

void starpu_perfmodel_directory(FILE *output)
{
	fprintf(output, "directory: <%s>\n", _starpu_get_perf_model_dir_codelet());
}

/* This function is intended to be used by external tools that should read
 * the performance model files */
int starpu_perfmodel_list(FILE *output)
{
#ifdef HAVE_SCANDIR
        char *path;
	struct dirent **list;
	int n;

	path = _starpu_get_perf_model_dir_codelet();
	n = scandir(path, &list, NULL, alphasort);
	if (n < 0)
	{
		_STARPU_DISP("Could not open the perfmodel directory <%s>: %s\n", path, strerror(errno));
		return 1;
	}
	else
	{
		int i;
		for (i = 0; i < n; i++)
		{
			if (strcmp(list[i]->d_name, ".") && strcmp(list[i]->d_name, ".."))
				fprintf(output, "file: <%s>\n", list[i]->d_name);
			free(list[i]);
		}
		free(list);
		return 0;
	}
#else
	_STARPU_MSG("Listing perfmodels is not implemented on pure Windows yet\n");
	return 1;
#endif
}

/* This function is intended to be used by external tools that should read the
 * performance model files */
/* TODO: write an clear function, to free symbol and history */
int starpu_perfmodel_load_symbol(const char *symbol, struct starpu_perfmodel *model)
{
	model->symbol = strdup(symbol);

	/* where is the file if it exists ? */
	char path[STR_LONG_LENGTH];
	starpu_perfmodel_get_model_path(model->symbol, path, sizeof(path));

	//	_STARPU_DEBUG("get_model_path -> %s\n", path);

	/* does it exist ? */
	int res;
	res = access(path, F_OK);
	if (res)
	{
		const char *dot = strrchr(symbol, '.');
		if (dot)
		{
			char *symbol2 = strdup(symbol);
			symbol2[dot-symbol] = '\0';
			int ret;
			_STARPU_DISP("note: loading history from %s instead of %s\n", symbol2, symbol);
			ret = starpu_perfmodel_load_symbol(symbol2,model);
			free(symbol2);
			return ret;
		}
		_STARPU_DISP("There is no performance model for symbol %s\n", symbol);
		return 1;
	}

	return starpu_perfmodel_load_file(path, model);
}

int starpu_perfmodel_load_file(const char *filename, struct starpu_perfmodel *model)
{
	int res, ret = 0;
	FILE *f = fopen(filename, "r");
	int locked;

	STARPU_ASSERT(f);

	starpu_perfmodel_init(model);

	locked = _starpu_frdlock(f) == 0;
	ret = parse_model_file(f, filename, model, 1);
	if (locked)
		_starpu_frdunlock(f);

	res = fclose(f);
	STARPU_ASSERT(res == 0);

	if (ret)
		starpu_perfmodel_unload_model(model);
	else
		model->is_loaded = 1;
	return ret;
}

int starpu_perfmodel_unload_model(struct starpu_perfmodel *model)
{
	if (model->symbol)
	{
		free((char *)model->symbol);
		model->symbol = NULL;
	}

	_starpu_deinitialize_performance_model(model);
	free(model->state);
	model->state = NULL;

	STARPU_PTHREAD_RWLOCK_WRLOCK(&registered_models_rwlock);
	struct _starpu_perfmodel *node;
	for (node  = _starpu_perfmodel_list_begin(&registered_models);
	     node != _starpu_perfmodel_list_end(&registered_models);
	     node  = _starpu_perfmodel_list_next(node))
	{
		if (node->model == model)
		{
			_starpu_perfmodel_list_erase(&registered_models, node);
			_starpu_perfmodel_delete(node);
			break;
		}
	}
	STARPU_PTHREAD_RWLOCK_UNLOCK(&registered_models_rwlock);

	return 0;
}

char* starpu_perfmodel_get_archtype_name(enum starpu_worker_archtype archtype)
{
	switch(archtype)
	{
		case(STARPU_CPU_WORKER):
			return "cpu";
			break;
		case(STARPU_CUDA_WORKER):
			return "cuda";
			break;
		case(STARPU_OPENCL_WORKER):
			return "opencl";
			break;
		case(STARPU_MIC_WORKER):
			return "mic";
			break;
		case(STARPU_MPI_MS_WORKER):
			return "mpi_ms";
			break;
		default:
			STARPU_ABORT();
			break;
	}
}

void starpu_perfmodel_get_arch_name(struct starpu_perfmodel_arch* arch, char *archname, size_t maxlen,unsigned impl)
{
	int i;
	int comb = _starpu_perfmodel_create_comb_if_needed(arch);

	STARPU_ASSERT(comb != -1);
	char devices[STR_VERY_LONG_LENGTH];
	int written = 0;
	devices[0] = '\0';
	for(i=0 ; i<arch->ndevices ; i++)
	{
		written += snprintf(devices + written, sizeof(devices)-written, "%s%d%s", starpu_perfmodel_get_archtype_name(arch->devices[i].type), arch->devices[i].devid, i != arch->ndevices-1 ? "_":"");
	}
	snprintf(archname, maxlen, "%s_impl%u (Comb%d)", devices, impl, comb);
}

void starpu_perfmodel_debugfilepath(struct starpu_perfmodel *model,
				    struct starpu_perfmodel_arch* arch, char *path, size_t maxlen, unsigned nimpl)
{
	int comb = starpu_perfmodel_arch_comb_get(arch->ndevices, arch->devices);
	STARPU_ASSERT(comb != -1);
	char archname[STR_SHORT_LENGTH];
	starpu_perfmodel_get_arch_name(arch, archname, sizeof(archname), nimpl);

	STARPU_ASSERT(path);

	get_model_debug_path(model, archname, path, maxlen);
}

double _starpu_regression_based_job_expected_perf(struct starpu_perfmodel *model, struct starpu_perfmodel_arch* arch, struct _starpu_job *j, unsigned nimpl)
{
	int comb;
	double exp = NAN;
	size_t size;
	struct starpu_perfmodel_regression_model *regmodel = NULL;

	comb = starpu_perfmodel_arch_comb_get(arch->ndevices, arch->devices);
	size = _starpu_job_get_data_size(model, arch, nimpl, j);

	if(comb == -1)
		goto docal;
	if (model->state->per_arch[comb] == NULL)
		// The model has not been executed on this combination
		goto docal;

	regmodel = &model->state->per_arch[comb][nimpl].regression;

	if (regmodel->valid && size >= regmodel->minx * 0.9 && size <= regmodel->maxx * 1.1)
                exp = regmodel->alpha*pow((double)size, regmodel->beta);

docal:
	STARPU_HG_DISABLE_CHECKING(model->benchmarking);
	if (isnan(exp) && !model->benchmarking)
	{
		char archname[STR_SHORT_LENGTH];

		starpu_perfmodel_get_arch_name(arch, archname, sizeof(archname), nimpl);
		_STARPU_DISP("Warning: model %s is not calibrated enough for %s size %lu (only %u measurements from size %lu to %lu), forcing calibration for this run. Use the STARPU_CALIBRATE environment variable to control this. You probably need to run again to continue calibrating the model, until this warning disappears.\n", model->symbol, archname, (unsigned long) size, regmodel?regmodel->nsample:0, regmodel?regmodel->minx:0, regmodel?regmodel->maxx:0);
		_starpu_set_calibrate_flag(1);
		model->benchmarking = 1;
	}

	return exp;
}

double _starpu_non_linear_regression_based_job_expected_perf(struct starpu_perfmodel *model, struct starpu_perfmodel_arch* arch, struct _starpu_job *j,unsigned nimpl)
{
	int comb;
	double exp = NAN;
	size_t size;
	struct starpu_perfmodel_regression_model *regmodel;
	struct starpu_perfmodel_history_table *entry = NULL;

	size = _starpu_job_get_data_size(model, arch, nimpl, j);
	comb = starpu_perfmodel_arch_comb_get(arch->ndevices, arch->devices);
	if(comb == -1)
		goto docal;
	if (model->state->per_arch[comb] == NULL)
		// The model has not been executed on this combination
		goto docal;

	regmodel = &model->state->per_arch[comb][nimpl].regression;

	if (regmodel->nl_valid && size >= regmodel->minx * 0.9 && size <= regmodel->maxx * 1.1)
		exp = regmodel->a*pow((double)size, regmodel->b) + regmodel->c;
	else
	{
		uint32_t key = _starpu_compute_buffers_footprint(model, arch, nimpl, j);
		struct starpu_perfmodel_per_arch *per_arch_model = &model->state->per_arch[comb][nimpl];
		struct starpu_perfmodel_history_table *history;

		STARPU_PTHREAD_RWLOCK_RDLOCK(&model->state->model_rwlock);
		history = per_arch_model->history;
		HASH_FIND_UINT32_T(history, &key, entry);
		STARPU_PTHREAD_RWLOCK_UNLOCK(&model->state->model_rwlock);

		/* Here helgrind would shout that this is unprotected access.
		 * We do not care about racing access to the mean, we only want
		 * a good-enough estimation */

		if (entry && entry->history_entry && entry->history_entry->nsample >= _starpu_calibration_minimum)
			exp = entry->history_entry->mean;

docal:
		STARPU_HG_DISABLE_CHECKING(model->benchmarking);
		if (isnan(exp) && !model->benchmarking)
		{
			char archname[STR_SHORT_LENGTH];

			starpu_perfmodel_get_arch_name(arch, archname, sizeof(archname), nimpl);
			_STARPU_DISP("Warning: model %s is not calibrated enough for %s size %lu (only %u measurements), forcing calibration for this run. Use the STARPU_CALIBRATE environment variable to control this. You probably need to run again to continue calibrating the model, until this warning disappears.\n", model->symbol, archname, (unsigned long) size, entry && entry->history_entry ? entry->history_entry->nsample : 0);
			_starpu_set_calibrate_flag(1);
			model->benchmarking = 1;
		}
	}

	return exp;
}

double _starpu_multiple_regression_based_job_expected_perf(struct starpu_perfmodel *model, struct starpu_perfmodel_arch* arch, struct _starpu_job *j, unsigned nimpl)
{
	int comb;
	double expected_duration=NAN;

	struct starpu_perfmodel_regression_model *reg_model = NULL;
	comb = starpu_perfmodel_arch_comb_get(arch->ndevices, arch->devices);
	if(comb == -1)
		goto docal;
	if (model->state->per_arch[comb] == NULL)
		// The model has not been executed on this combination
		goto docal;
	reg_model = &model->state->per_arch[comb][nimpl].regression;
	if (reg_model->coeff == NULL)
		goto docal;

	double *parameters;
	_STARPU_MALLOC(parameters, model->nparameters*sizeof(double));
	model->parameters(j->task, parameters);
	expected_duration=reg_model->coeff[0];
	unsigned i;
	for (i=0; i < model->ncombinations; i++)
	{
		double parameter_value=1.;
		unsigned k;
		for (k=0; k < model->nparameters; k++)
			parameter_value *= pow(parameters[k],model->combinations[i][k]);

		expected_duration += reg_model->coeff[i+1]*parameter_value;
	}

docal:
	STARPU_HG_DISABLE_CHECKING(model->benchmarking);
	if (isnan(expected_duration) && !model->benchmarking)
	{
		char archname[STR_SHORT_LENGTH];

		starpu_perfmodel_get_arch_name(arch, archname, sizeof(archname), nimpl);
		_STARPU_DISP("Warning: model %s is not calibrated enough for %s, forcing calibration for this run. Use the STARPU_CALIBRATE environment variable to control this. You probably need to run again to continue calibrating the model, until this warning disappears.\n", model->symbol, archname);
		_starpu_set_calibrate_flag(1);
		model->benchmarking = 1;
	}

	// In the unlikely event that predicted duration is negative
	// in case multiple linear regression is not so accurate
	if (expected_duration < 0 )
		expected_duration = 0.00001;

	//Make sure that the injected time is in milliseconds
	return expected_duration;
}

double _starpu_history_based_job_expected_perf(struct starpu_perfmodel *model, struct starpu_perfmodel_arch* arch, struct _starpu_job *j,unsigned nimpl)
{
	int comb;
	double exp = NAN;
	struct starpu_perfmodel_per_arch *per_arch_model;
	struct starpu_perfmodel_history_entry *entry = NULL;
	struct starpu_perfmodel_history_table *history, *elt;
	uint32_t key;

	comb = starpu_perfmodel_arch_comb_get(arch->ndevices, arch->devices);
	key = _starpu_compute_buffers_footprint(model, arch, nimpl, j);
	if(comb == -1)
		goto docal;
	if (model->state->per_arch[comb] == NULL)
		// The model has not been executed on this combination
		goto docal;

	per_arch_model = &model->state->per_arch[comb][nimpl];

	STARPU_PTHREAD_RWLOCK_RDLOCK(&model->state->model_rwlock);
	history = per_arch_model->history;
	HASH_FIND_UINT32_T(history, &key, elt);
	entry = (elt == NULL) ? NULL : elt->history_entry;
	STARPU_ASSERT_MSG(!entry || entry->mean >= 0, "entry=%p, entry->mean=%lf\n", entry, entry?entry->mean:NAN);
	STARPU_PTHREAD_RWLOCK_UNLOCK(&model->state->model_rwlock);

	/* Here helgrind would shout that this is unprotected access.
	 * We do not care about racing access to the mean, we only want
	 * a good-enough estimation */

	if (entry && entry->nsample)
	{
#ifdef STARPU_SIMGRID
		if (entry->nsample < _starpu_calibration_minimum)
		{
			char archname[STR_SHORT_LENGTH];
			starpu_perfmodel_get_arch_name(arch, archname, sizeof(archname), nimpl);

			_STARPU_DISP("Warning: model %s is not calibrated enough for %s size %ld footprint %x (only %u measurements). Using it anyway for the simulation\n", model->symbol, archname, j->task?(long int)_starpu_job_get_data_size(model, arch, nimpl, j):-1, key, entry->nsample);
		}
#else
		if (entry->nsample >= _starpu_calibration_minimum)
#endif
		{
			STARPU_ASSERT_MSG(entry->mean >= 0, "entry->mean=%lf\n", entry->mean);
			/* TODO: report differently if we've scheduled really enough
			 * of that task and the scheduler should perhaps put it aside */
			/* Calibrated enough */
			exp = entry->mean;
		}
	}

docal:
#ifdef STARPU_SIMGRID
	if (isnan(exp))
	{
		char archname[STR_SHORT_LENGTH];
		starpu_perfmodel_get_arch_name(arch, archname, sizeof(archname), nimpl);

		_STARPU_DISP("Warning: model %s is not calibrated at all for %s size %ld footprint %x. Assuming it can not work there\n", model->symbol, archname, j->task?(long int)_starpu_job_get_data_size(model, arch, nimpl, j):-1, key);
		exp = 0.;
	}
#else
	STARPU_HG_DISABLE_CHECKING(model->benchmarking);
	if (isnan(exp) && !model->benchmarking)
	{
		char archname[STR_SHORT_LENGTH];

		starpu_perfmodel_get_arch_name(arch, archname, sizeof(archname), nimpl);
		_STARPU_DISP("Warning: model %s is not calibrated enough for %s size %ld footprint %x (only %u measurements), forcing calibration for this run. Use the STARPU_CALIBRATE environment variable to control this. You probably need to run again to continue calibrating the model, until this warning disappears.\n", model->symbol, archname, j->task?(long int)_starpu_job_get_data_size(model, arch, nimpl, j):-1, key, entry ? entry->nsample : 0);
		_starpu_set_calibrate_flag(1);
		model->benchmarking = 1;
	}
#endif

	STARPU_ASSERT_MSG(isnan(exp)||exp >= 0, "exp=%lf\n", exp);
	return exp;
}

double starpu_perfmodel_history_based_expected_perf(struct starpu_perfmodel *model, struct starpu_perfmodel_arch * arch, uint32_t footprint)
{
	struct _starpu_job j =
		{
			.footprint = footprint,
			.footprint_is_computed = 1,
		};
	return _starpu_history_based_job_expected_perf(model, arch, &j, j.nimpl);
}

int _starpu_perfmodel_create_comb_if_needed(struct starpu_perfmodel_arch* arch)
{
	int comb = starpu_perfmodel_arch_comb_get(arch->ndevices, arch->devices);
	if(comb == -1)
		comb = starpu_perfmodel_arch_comb_add(arch->ndevices, arch->devices);
	return comb;
}

void _starpu_update_perfmodel_history(struct _starpu_job *j, struct starpu_perfmodel *model, struct starpu_perfmodel_arch* arch, unsigned cpuid STARPU_ATTRIBUTE_UNUSED, double measured, unsigned impl)
{
	STARPU_ASSERT_MSG(measured >= 0, "measured=%lf\n", measured);
	if (model)
	{
		int c;
		unsigned found = 0;
		int comb = _starpu_perfmodel_create_comb_if_needed(arch);

		STARPU_PTHREAD_RWLOCK_WRLOCK(&model->state->model_rwlock);

		for(c = 0; c < model->state->ncombs; c++)
		{
			if(model->state->combs[c] == comb)
			{
				found = 1;
				break;
			}
		}

		if(!found)
		{
			if (model->state->ncombs + 1 >= model->state->ncombs_set)
			{
				// The number of combinations is bigger than the one which was initially allocated, we need to reallocate,
				// do not only reallocate 1 extra comb, rather reallocate 5 to avoid too frequent calls to _starpu_perfmodel_realloc
				_starpu_perfmodel_realloc(model, model->state->ncombs_set+5);
			}
			model->state->combs[model->state->ncombs++] = comb;
		}

		if(!model->state->per_arch[comb])
		{
			_starpu_perfmodel_malloc_per_arch(model, comb, STARPU_MAXIMPLEMENTATIONS);
			_starpu_perfmodel_malloc_per_arch_is_set(model, comb, STARPU_MAXIMPLEMENTATIONS);
		}

		struct starpu_perfmodel_per_arch *per_arch_model = &model->state->per_arch[comb][impl];
		if (model->state->per_arch_is_set[comb][impl] == 0)
		{
			// We are adding a new implementation for the given comb and the given impl
			model->state->nimpls[comb]++;
			model->state->per_arch_is_set[comb][impl] = 1;
		}

		if (model->type == STARPU_HISTORY_BASED || model->type == STARPU_NL_REGRESSION_BASED || model->type == STARPU_REGRESSION_BASED)
		{
			struct starpu_perfmodel_history_entry *entry;
			struct starpu_perfmodel_history_table *elt;
			struct starpu_perfmodel_history_list **list;
			uint32_t key = _starpu_compute_buffers_footprint(model, arch, impl, j);

			list = &per_arch_model->list;

			HASH_FIND_UINT32_T(per_arch_model->history, &key, elt);
			entry = (elt == NULL) ? NULL : elt->history_entry;

			if (!entry)
			{
				/* this is the first entry with such a footprint */
				_STARPU_CALLOC(entry, 1, sizeof(struct starpu_perfmodel_history_entry));

				/* Tell  helgrind that we do not care about
				 * racing access to the sampling, we only want a
				 * good-enough estimation */
				STARPU_HG_DISABLE_CHECKING(entry->nsample);
				STARPU_HG_DISABLE_CHECKING(entry->mean);

				/* For history-based, do not take the first measurement into account, it is very often quite bogus */
				/* TODO: it'd be good to use a better estimation heuristic, like the median, or latest n values, etc. */
				if (model->type != STARPU_HISTORY_BASED) {
					entry->sum = measured;
					entry->sum2 = measured*measured;
					entry->nsample = 1;
					entry->mean = measured;
				}

				entry->size = _starpu_job_get_data_size(model, arch, impl, j);
				entry->flops = j->task->flops;

				entry->footprint = key;

				insert_history_entry(entry, list, &per_arch_model->history);
			}
			else
			{
				/* There is already an entry with the same footprint */

				double local_deviation = measured/entry->mean;

				if (entry->nsample &&
					(100 * local_deviation > (100 + historymaxerror)
					 || (100 / local_deviation > (100 + historymaxerror))))
				{
					entry->nerror++;

					/* More errors than measurements, we're most probably completely wrong, we flush out all the entries */
					if (entry->nerror >= entry->nsample)
					{
						char archname[STR_SHORT_LENGTH];
						starpu_perfmodel_get_arch_name(arch, archname, sizeof(archname), impl);
						_STARPU_DISP("Too big deviation for model %s on %s: %fus vs average %fus, %u such errors against %u samples (%+f%%), flushing the performance model. Use the STARPU_HISTORY_MAX_ERROR environement variable to control the threshold (currently %d%%)\n", model->symbol, archname, measured, entry->mean, entry->nerror, entry->nsample, measured * 100. / entry->mean - 100, historymaxerror);
						entry->sum = 0.0;
						entry->sum2 = 0.0;
						entry->nsample = 0;
						entry->nerror = 0;
						entry->mean = 0.0;
						entry->deviation = 0.0;
					}
				}
				else
				{
					entry->sum += measured;
					entry->sum2 += measured*measured;
					entry->nsample++;

					unsigned n = entry->nsample;
					entry->mean = entry->sum / n;
					entry->deviation = sqrt((fabs(entry->sum2 - (entry->sum*entry->sum)/n))/n);
				}

				if (j->task->flops != 0. && !isnan(entry->flops))
				{
					if (entry->flops == 0.)
						entry->flops = j->task->flops;
					else if ((fabs(entry->flops - j->task->flops) / entry->flops) > 0.00001)
					{
						/* Incoherent flops! forget about trying to record flops */
						_STARPU_DISP("Incoherent flops in model %s: %f vs previous %f, stopping recording flops\n", model->symbol, j->task->flops, entry->flops);
						entry->flops = NAN;
					}
				}
			}

			STARPU_ASSERT(entry);
		}

		if (model->type == STARPU_REGRESSION_BASED || model->type == STARPU_NL_REGRESSION_BASED)
		{
			struct starpu_perfmodel_regression_model *reg_model;
			reg_model = &per_arch_model->regression;

			/* update the regression model */
			size_t job_size = _starpu_job_get_data_size(model, arch, impl, j);
			double logy, logx;
			logx = log((double)job_size);
			logy = log(measured);

			reg_model->sumlnx += logx;
			reg_model->sumlnx2 += logx*logx;
			reg_model->sumlny += logy;
			reg_model->sumlnxlny += logx*logy;
			if (reg_model->minx == 0 || job_size < reg_model->minx)
				reg_model->minx = job_size;
			if (reg_model->maxx == 0 || job_size > reg_model->maxx)
				reg_model->maxx = job_size;
			reg_model->nsample++;

			if (VALID_REGRESSION(reg_model))
			{
				unsigned n = reg_model->nsample;

				double num = (n*reg_model->sumlnxlny - reg_model->sumlnx*reg_model->sumlny);
				double denom = (n*reg_model->sumlnx2 - reg_model->sumlnx*reg_model->sumlnx);

				reg_model->beta = num/denom;
				reg_model->alpha = exp((reg_model->sumlny - reg_model->beta*reg_model->sumlnx)/n);
				reg_model->valid = 1;
			}
		}

		if (model->type == STARPU_MULTIPLE_REGRESSION_BASED)
		{
			struct starpu_perfmodel_history_entry *entry;
			struct starpu_perfmodel_history_list **list;
			list = &per_arch_model->list;

			_STARPU_CALLOC(entry, 1, sizeof(struct starpu_perfmodel_history_entry));
			_STARPU_MALLOC(entry->parameters, model->nparameters*sizeof(double));
			model->parameters(j->task, entry->parameters);
			entry->tag = j->task->tag_id;
			STARPU_ASSERT(measured >= 0);
			entry->duration = measured;

			struct starpu_perfmodel_history_list *link;
			_STARPU_MALLOC(link, sizeof(struct starpu_perfmodel_history_list));
			link->next = *list;
			link->entry = entry;
			*list = link;
		}

#ifdef STARPU_MODEL_DEBUG
		struct starpu_task *task = j->task;
		starpu_perfmodel_debugfilepath(model, arch_combs[comb], per_arch_model->debug_path, STR_LONG_LENGTH, impl);
		FILE *f = fopen(per_arch_model->debug_path, "a+");
		int locked;
		if (f == NULL)
		{
			_STARPU_DISP("Error <%s> when opening file <%s>\n", strerror(errno), per_arch_model->debug_path);
			STARPU_PTHREAD_RWLOCK_UNLOCK(&model->state->model_rwlock);
			return;
		}
		locked = _starpu_fwrlock(f) == 0;

		if (!j->footprint_is_computed)
			(void) _starpu_compute_buffers_footprint(model, arch, impl, j);

		STARPU_ASSERT(j->footprint_is_computed);

		fprintf(f, "0x%x\t%lu\t%f\t%f\t%f\t%u\t\t", j->footprint, (unsigned long) _starpu_job_get_data_size(model, arch, impl, j), measured, task->predicted, task->predicted_transfer, cpuid);
		unsigned i;
		unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(task);

		for (i = 0; i < nbuffers; i++)
		{
			starpu_data_handle_t handle = STARPU_TASK_GET_HANDLE(task, i);

			STARPU_ASSERT(handle->ops);
			STARPU_ASSERT(handle->ops->display);
			handle->ops->display(handle, f);
		}
		fprintf(f, "\n");
		if (locked)
			_starpu_fwrunlock(f);
		fclose(f);
#endif
		STARPU_PTHREAD_RWLOCK_UNLOCK(&model->state->model_rwlock);
	}
}

void starpu_perfmodel_update_history(struct starpu_perfmodel *model, struct starpu_task *task, struct starpu_perfmodel_arch * arch, unsigned cpuid, unsigned nimpl, double measured)
{
	struct _starpu_job *job = _starpu_get_job_associated_to_task(task);

#ifdef STARPU_SIMGRID
	STARPU_ASSERT_MSG(0, "We are not supposed to update history when simulating execution");
#endif

	_starpu_init_and_load_perfmodel(model);
	/* Record measurement */
	_starpu_update_perfmodel_history(job, model, arch, cpuid, measured, nimpl);
	/* and save perfmodel on termination */
	_starpu_set_calibrate_flag(1);
}

int starpu_perfmodel_list_combs(FILE *output, struct starpu_perfmodel *model)
{
	int comb;

	fprintf(output, "Model <%s>\n", model->symbol);
	for(comb = 0; comb < model->state->ncombs; comb++)
	{
		struct starpu_perfmodel_arch *arch;
		int device;

		arch = starpu_perfmodel_arch_comb_fetch(model->state->combs[comb]);
		fprintf(output, "\tComb %d: %d device%s\n", model->state->combs[comb], arch->ndevices, arch->ndevices>1?"s":"");
		for(device=0 ; device<arch->ndevices ; device++)
		{
			char *name = starpu_perfmodel_get_archtype_name(arch->devices[device].type);
			fprintf(output, "\t\tDevice %d: type: %s - devid: %d - ncores: %d\n", device, name, arch->devices[device].devid, arch->devices[device].ncores);
		}
	}
	return 0;
}

struct starpu_perfmodel_per_arch *starpu_perfmodel_get_model_per_arch(struct starpu_perfmodel *model, struct starpu_perfmodel_arch *arch, unsigned impl)
{
	int comb = starpu_perfmodel_arch_comb_get(arch->ndevices, arch->devices);
	if (comb == -1)
		return NULL;

	if (!model->state->per_arch[comb])
		return NULL;

	return &model->state->per_arch[comb][impl];
}

static struct starpu_perfmodel_per_arch *_starpu_perfmodel_get_model_per_devices(struct starpu_perfmodel *model, int impl, va_list varg_list)
{
	struct starpu_perfmodel_arch arch;
	va_list varg_list_copy;
	int i, arg_type;
	int is_cpu_set = 0;

	// We first count the number of devices
	arch.ndevices = 0;
	va_copy(varg_list_copy, varg_list);
	while ((arg_type = va_arg(varg_list_copy, int)) != -1)
	{
		int devid = va_arg(varg_list_copy, int);
		int ncores = va_arg(varg_list_copy, int);

		arch.ndevices ++;
		if (arg_type == STARPU_CPU_WORKER)
		{
			STARPU_ASSERT_MSG(is_cpu_set == 0, "STARPU_CPU_WORKER can only be specified once\n");
			STARPU_ASSERT_MSG(devid==0, "STARPU_CPU_WORKER must be followed by a value 0 for the device id");
			is_cpu_set = 1;
		}
		else
		{
			STARPU_ASSERT_MSG(ncores==1, "%s must be followed by a value 1 for ncores", starpu_worker_get_type_as_string(arg_type));
		}
	}
	va_end(varg_list_copy);

	// We set the devices
	_STARPU_MALLOC(arch.devices, arch.ndevices * sizeof(struct starpu_perfmodel_device));
	va_copy(varg_list_copy, varg_list);
	for(i=0 ; i<arch.ndevices ; i++)
	{
		arch.devices[i].type = va_arg(varg_list_copy, int);
		arch.devices[i].devid = va_arg(varg_list_copy, int);
		arch.devices[i].ncores = va_arg(varg_list_copy, int);
	}
	va_end(varg_list_copy);

	// Get the combination for this set of devices
	int comb = _starpu_perfmodel_create_comb_if_needed(&arch);

	free(arch.devices);

	// Realloc if necessary
	if (comb >= model->state->ncombs_set)
		_starpu_perfmodel_realloc(model, comb+1);

	// Get the per_arch object
	if (model->state->per_arch[comb] == NULL)
	{
		_starpu_perfmodel_malloc_per_arch(model, comb, STARPU_MAXIMPLEMENTATIONS);
		_starpu_perfmodel_malloc_per_arch_is_set(model, comb, STARPU_MAXIMPLEMENTATIONS);
		model->state->nimpls[comb] = 0;
	}
	model->state->per_arch_is_set[comb][impl] = 1;
	model->state->nimpls[comb] ++;

	return &model->state->per_arch[comb][impl];
}

struct starpu_perfmodel_per_arch *starpu_perfmodel_get_model_per_devices(struct starpu_perfmodel *model, int impl, ...)
{
	va_list varg_list;
	struct starpu_perfmodel_per_arch *per_arch;

	va_start(varg_list, impl);
	per_arch = _starpu_perfmodel_get_model_per_devices(model, impl, varg_list);
	va_end(varg_list);

	return per_arch;
}

int starpu_perfmodel_set_per_devices_cost_function(struct starpu_perfmodel *model, int impl, starpu_perfmodel_per_arch_cost_function func, ...)
{
	va_list varg_list;
	struct starpu_perfmodel_per_arch *per_arch;

	va_start(varg_list, func);
	per_arch = _starpu_perfmodel_get_model_per_devices(model, impl, varg_list);
	per_arch->cost_function = func;
	va_end(varg_list);

	return 0;
}

int starpu_perfmodel_set_per_devices_size_base(struct starpu_perfmodel *model, int impl, starpu_perfmodel_per_arch_size_base func, ...)
{
	va_list varg_list;
	struct starpu_perfmodel_per_arch *per_arch;

	va_start(varg_list, func);
	per_arch = _starpu_perfmodel_get_model_per_devices(model, impl, varg_list);
	per_arch->size_base = func;
	va_end(varg_list);

	return 0;
}
