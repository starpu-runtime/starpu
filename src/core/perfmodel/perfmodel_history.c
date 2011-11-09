/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010, 2011  Université de Bordeaux 1
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
 * Copyright (C) 2011  Télécom-SudParis
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

#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>
#include <common/config.h>
#include <core/perfmodel/perfmodel.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <pthread.h>
#include <datawizard/datawizard.h>
#include <core/perfmodel/regression.h>
#include <common/config.h>
#include <starpu_parameters.h>

#ifdef STARPU_HAVE_WINDOWS
#include <windows.h>
#endif
		
static pthread_rwlock_t registered_models_rwlock;
static struct starpu_model_list_t *registered_models = NULL;

/*
 * History based model
 */


static void insert_history_entry(struct starpu_history_entry_t *entry, struct starpu_history_list_t **list, struct starpu_htbl32_node_s **history_ptr)
{
	struct starpu_history_list_t *link;
	struct starpu_history_entry_t *old;

	link = (struct starpu_history_list_t *) malloc(sizeof(struct starpu_history_list_t));
	link->next = *list;
	link->entry = entry;
	*list = link;

	old = (struct starpu_history_entry_t *) _starpu_htbl_insert_32(history_ptr, entry->footprint, entry);
	/* that may fail in case there is some concurrency issue */
	STARPU_ASSERT(old == NULL);
}


static void dump_reg_model(FILE *f, struct starpu_perfmodel_t *model, unsigned arch, unsigned nimpl)
{
	struct starpu_per_arch_perfmodel_t *per_arch_model;

	per_arch_model = &model->per_arch[arch][nimpl];
	struct starpu_regression_model_t *reg_model;
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

	fprintf(f, "# sumlnx\tsumlnx2\t\tsumlny\t\tsumlnxlny\talpha\t\tbeta\t\tn\n");
	fprintf(f, "%-15le\t%-15le\t%-15le\t%-15le\t%-15le\t%-15le\t%u\n", reg_model->sumlnx, reg_model->sumlnx2, reg_model->sumlny, reg_model->sumlnxlny, alpha, beta, reg_model->nsample);

	/*
	 * Non-Linear Regression model
	 */

	double a = nan(""), b = nan(""), c = nan("");

	if (model->type == STARPU_NL_REGRESSION_BASED)
		_starpu_regression_non_linear_power(per_arch_model->list, &a, &b, &c);

	fprintf(f, "# a\t\tb\t\tc\n");
	fprintf(f, "%-15le\t%-15le\t%-15le\n", a, b, c);
}

static void scan_reg_model(FILE *f, struct starpu_regression_model_t *reg_model)
{
	int res;

	/*
	 * Linear Regression model
	 */

	_starpu_drop_comments(f);

	res = fscanf(f, "%le\t%le\t%le\t%le\t%le\t%le\t%u\n",
		&reg_model->sumlnx, &reg_model->sumlnx2, &reg_model->sumlny,
		&reg_model->sumlnxlny, &reg_model->alpha, &reg_model->beta,
		&reg_model->nsample);
	STARPU_ASSERT(res == 7);

	/* If any of the parameters describing the linear regression model is NaN, the model is invalid */
	unsigned invalid = (isnan(reg_model->alpha)||isnan(reg_model->beta));
	reg_model->valid = !invalid && reg_model->nsample >= STARPU_CALIBRATION_MINIMUM;

	/*
	 * Non-Linear Regression model
	 */

	_starpu_drop_comments(f);

	res = fscanf(f, "%le\t%le\t%le\n", &reg_model->a, &reg_model->b, &reg_model->c);
	STARPU_ASSERT(res == 3);

	/* If any of the parameters describing the non-linear regression model is NaN, the model is invalid */
	unsigned nl_invalid = (isnan(reg_model->a)||isnan(reg_model->b)||isnan(reg_model->c));
	reg_model->nl_valid = !nl_invalid && reg_model->nsample >= STARPU_CALIBRATION_MINIMUM;
}

static void dump_history_entry(FILE *f, struct starpu_history_entry_t *entry)
{
	fprintf(f, "%08x\t%-15lu\t%-15le\t%-15le\t%-15le\t%-15le\t%u\n", entry->footprint, (unsigned long) entry->size, entry->mean, entry->deviation, entry->sum, entry->sum2, entry->nsample);
}

static void scan_history_entry(FILE *f, struct starpu_history_entry_t *entry)
{
	int res;

	_starpu_drop_comments(f);

	/* In case entry is NULL, we just drop these values */
	unsigned nsample;
	uint32_t footprint;
#ifdef STARPU_HAVE_WINDOWS
	unsigned size; /* in bytes */
#else
	size_t size; /* in bytes */
#endif
	double mean;
	double deviation;
	double sum;
	double sum2;

	/* Read the values from the file */
	res = fscanf(f, "%x\t%"
#ifndef STARPU_HAVE_WINDOWS
	"z"
#endif
	"u\t%le\t%le\t%le\t%le\t%u\n", &footprint, &size, &mean, &deviation, &sum, &sum2, &nsample);
	STARPU_ASSERT(res == 7);

	if (entry)
	{
		entry->footprint = footprint;
		entry->size = size;
		entry->mean = mean;
		entry->deviation = deviation;
		entry->sum = sum;
		entry->sum2 = sum2;
		entry->nsample = nsample;
	}
}

static void parse_per_arch_model_file(FILE *f, struct starpu_per_arch_perfmodel_t *per_arch_model, unsigned scan_history)
{
	unsigned nentries;

	_starpu_drop_comments(f);

	int res = fscanf(f, "%u\n", &nentries);
	STARPU_ASSERT(res == 1);

	scan_reg_model(f, &per_arch_model->regression);

	/* parse cpu entries */
	unsigned i;
	for (i = 0; i < nentries; i++) {
		struct starpu_history_entry_t *entry = NULL;
		if (scan_history)
		{
			entry = (struct starpu_history_entry_t *) malloc(sizeof(struct starpu_history_entry_t));
			STARPU_ASSERT(entry);
		}

		scan_history_entry(f, entry);
		
		/* insert the entry in the hashtable and the list structures  */
		if (scan_history)
			insert_history_entry(entry, &per_arch_model->list, &per_arch_model->history);
	}
}

static void parse_arch(FILE *f, struct starpu_perfmodel_t *model, unsigned scan_history, unsigned archmin, unsigned archmax, int skiparch)
{
	struct starpu_per_arch_perfmodel_t dummy;
	int nimpls, implmax, skipimpl, impl;
	unsigned ret, arch;
	

	for (arch = archmin; arch < archmax; arch++) {
		_starpu_drop_comments(f);
		ret = fscanf(f, "%d\n", &nimpls);
		STARPU_ASSERT(ret == 1);
		implmax = STARPU_MIN(nimpls, STARPU_MAXIMPLEMENTATIONS);
		skipimpl = nimpls - STARPU_MAXIMPLEMENTATIONS;
		for (impl = 0; impl < implmax; impl++) {
			parse_per_arch_model_file(f, &model->per_arch[arch][impl], scan_history);
		}
		if (skipimpl > 0) {
			for (impl = 0; impl < skipimpl; impl++) {
				parse_per_arch_model_file(f, &dummy, 0);
			}
		}
	}

	if (skiparch > 0) {
		_starpu_drop_comments(f);
		ret = fscanf(f, "%d\n", &nimpls);
		STARPU_ASSERT(ret == 1);
		implmax = STARPU_MIN(nimpls, STARPU_MAXIMPLEMENTATIONS);
		skipimpl = nimpls - STARPU_MAXIMPLEMENTATIONS;
		for (arch = 0; arch < skiparch; arch ++) {
			for (impl = 0; impl < implmax; impl++) {
				parse_per_arch_model_file(f, &dummy, 0);
			}
			if (skipimpl > 0) {
				for (impl = 0; impl < skipimpl; impl++) {
					parse_per_arch_model_file(f, &dummy, 0);
				}
			}
		}
	}
}

static void parse_model_file(FILE *f, struct starpu_perfmodel_t *model, unsigned scan_history)
{
	unsigned ret;
	unsigned archmin = 0;
	unsigned max_gordondevs = 1; /* XXX : we need a STARPU_MAXGORDONDEVS cst */
	unsigned narchs;

	/* We could probably write a clean loop here, but the code would not
	 * really be easier to read. */

	/* Parsing CPUs */
	_starpu_drop_comments(f);
	ret = fscanf(f, "%u\n", &narchs);
	STARPU_ASSERT(ret == 1);

	_STARPU_DEBUG("Parsing %u CPUs\n", narchs);
	if (narchs > 0)
	{
		parse_arch(f, model, scan_history,
				archmin,
				STARPU_MIN(narchs, STARPU_MAXCPUS),
				narchs - STARPU_MAXCPUS);
	}

	/* Parsing CUDA devs */
	_starpu_drop_comments(f);
	ret = fscanf(f, "%u\n", &narchs);
	STARPU_ASSERT(ret == 1);
	archmin += STARPU_MAXCPUS;
	_STARPU_DEBUG("Parsing %u CUDA devices\n", narchs);
	if (narchs > 0)
	{
		parse_arch(f, model, scan_history,
				archmin,
				archmin + STARPU_MIN(narchs, STARPU_MAXCUDADEVS),
				narchs - STARPU_MAXCUDADEVS);
	}

	/* Parsing OpenCL devs */
	_starpu_drop_comments(f);
	ret = fscanf(f, "%u\n", &narchs);
	STARPU_ASSERT(ret == 1);

	archmin += STARPU_MAXCUDADEVS;
	_STARPU_DEBUG("Parsing %u OpenCL devices\n", narchs);
	if (narchs > 0)
	{
		parse_arch(f, model, scan_history,
				archmin,
				archmin + STARPU_MIN(narchs, STARPU_MAXOPENCLDEVS),
				narchs - STARPU_MAXOPENCLDEVS);
	}

	/* Parsing Gordon implementations */
	_starpu_drop_comments(f);
	ret = fscanf(f, "%u\n", &narchs);
	STARPU_ASSERT(ret == 1);

	archmin += STARPU_MAXOPENCLDEVS;
	_STARPU_DEBUG("Parsing %u Gordon devices\n", narchs);
	if (narchs > 0)
	{
		parse_arch(f, model, scan_history,
				archmin,
				archmin + max_gordondevs,
				narchs - max_gordondevs);
	}
}


static void dump_per_arch_model_file(FILE *f, struct starpu_perfmodel_t *model, unsigned arch, unsigned nimpl)
{
	struct starpu_per_arch_perfmodel_t *per_arch_model;

	per_arch_model = &model->per_arch[arch][nimpl];
	/* count the number of elements in the lists */
	struct starpu_history_list_t *ptr = NULL;
	unsigned nentries = 0;

	if (model->type == STARPU_HISTORY_BASED || model->type == STARPU_NL_REGRESSION_BASED)
	{
		/* Dump the list of all entries in the history */
		ptr = per_arch_model->list;
		while(ptr) {
			nentries++;
			ptr = ptr->next;
		}
	}

	if (nentries == 0)
		return;
	/* header */
	char archname[32];
	starpu_perfmodel_get_arch_name((enum starpu_perf_archtype) arch, archname, 32, nimpl);
	fprintf(f, "# Model for %s\n", archname);
	fprintf(f, "# number of entries\n%u\n", nentries);

	dump_reg_model(f, model, arch, nimpl);

	/* Dump the history into the model file in case it is necessary */
	if (model->type == STARPU_HISTORY_BASED || model->type == STARPU_NL_REGRESSION_BASED)
	{
		fprintf(f, "# hash\t\tsize\t\tmean\t\tdev\t\tsum\t\tsum2\t\tn\n");
		ptr = per_arch_model->list;
		while (ptr) {
			dump_history_entry(f, ptr->entry);
			ptr = ptr->next;
		}
	}

	fprintf(f, "\n##################\n");
}

static unsigned get_n_entries(struct starpu_perfmodel_t *model, unsigned arch, unsigned impl)
{
	struct starpu_per_arch_perfmodel_t *per_arch_model;
	per_arch_model = &model->per_arch[arch][impl];
	/* count the number of elements in the lists */
	struct starpu_history_list_t *ptr = NULL;
	unsigned nentries = 0;

	if (model->type == STARPU_HISTORY_BASED || model->type == STARPU_NL_REGRESSION_BASED)
	{
		/* Dump the list of all entries in the history */
		ptr = per_arch_model->list;
		while(ptr) {
			nentries++;
			ptr = ptr->next;
		}
	}
	return nentries;
}

static void dump_model_file(FILE *f, struct starpu_perfmodel_t *model)
{
	unsigned number_of_archs[4] = { 0, 0, 0, 0};
	unsigned arch;
	unsigned nimpl;
	unsigned idx = 0;

	/* Finding the number of archs to write for each kind of device */
	for (arch = 0; arch < STARPU_NARCH_VARIATIONS; arch++)
	{
		switch (arch)
		{
			case STARPU_CUDA_DEFAULT:
			case STARPU_OPENCL_DEFAULT:
			case STARPU_GORDON_DEFAULT:
				idx++;
				break;
			default:
				break;
		}

		unsigned nentries = 0;
		for (nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
		{
			nentries = get_n_entries(model, arch, nimpl) != 0;
			if (nentries > 0)
			{
				number_of_archs[idx]++;
				break;
			}
		}
	}

	/* Writing stuff */
	char *name = "unknown";
	unsigned substract_to_arch = 0;
	for (arch = 0; arch < STARPU_NARCH_VARIATIONS; arch++)
	{
		switch (arch)
		{
			case STARPU_CPU_DEFAULT:
				name = "CPU";
				fprintf(f, "##################\n");
				fprintf(f, "# %ss\n", name);
				fprintf(f, "# number of %s architectures\n", name);
				fprintf(f, "%u\n", number_of_archs[0]);
				break;
			case STARPU_CUDA_DEFAULT:
				name = "CUDA";
				substract_to_arch = STARPU_MAXCPUS;
				fprintf(f, "##################\n");
				fprintf(f, "# %ss\n", name);
				fprintf(f, "# number of %s architectures\n", name);
				fprintf(f, "%u\n", number_of_archs[1]);
				break;
			case STARPU_OPENCL_DEFAULT:
				name = "OPENCL";
				substract_to_arch += STARPU_MAXCUDADEVS;
				fprintf(f, "##################\n");
				fprintf(f, "# %ss\n", name);
				fprintf(f, "# number of %s architectures\n", name);
				fprintf(f, "%u\n", number_of_archs[2]);
				break;
			case STARPU_GORDON_DEFAULT:
				name = "GORDON";
				substract_to_arch += STARPU_MAXOPENCLDEVS;
				fprintf(f, "##################\n");
				fprintf(f, "# %ss\n", name);
				fprintf(f, "# number of %s architectures\n", name);
				fprintf(f, "%u\n", number_of_archs[3]);
				break;
			default:
				break;
		}

		for (nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
		{
			if (get_n_entries(model, arch, nimpl) == 0)
				break;

		}
		unsigned max_impl = nimpl;

		if (max_impl == 0)
			continue;

		fprintf(f, "###########\n");
		fprintf(f, "# %s_%u\n", name, arch - substract_to_arch);
		fprintf(f, "# number of implementations\n");
		fprintf(f, "%u\n", max_impl);
		for (nimpl = 0; nimpl < max_impl; nimpl++)
		{
			dump_per_arch_model_file(f, model, arch, nimpl);
		}
	}
}

static void initialize_per_arch_model(struct starpu_per_arch_perfmodel_t *per_arch_model)
{
	per_arch_model->history = NULL;
	per_arch_model->list = NULL;
}

static void initialize_model(struct starpu_perfmodel_t *model)
{
	unsigned arch;
	unsigned nimpl;
	for (arch = 0; arch < STARPU_NARCH_VARIATIONS; arch++)
	{
		for (nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
		{
			initialize_per_arch_model(&model->per_arch[arch][nimpl]);
		}
	}
}

static void get_model_debug_path(struct starpu_perfmodel_t *model, const char *arch, char *path, size_t maxlen)
{
	STARPU_ASSERT(path);

	_starpu_get_perf_model_dir_debug(path, maxlen);
	strncat(path, model->symbol, maxlen);
	
	char hostname[32];
	char *forced_hostname = getenv("STARPU_HOSTNAME");
	if (forced_hostname && forced_hostname[0])
		snprintf(hostname, sizeof(hostname), "%s", forced_hostname);
	else
		gethostname(hostname, sizeof(hostname));
	strncat(path, ".", maxlen);
	strncat(path, hostname, maxlen);
	strncat(path, ".", maxlen);
	strncat(path, arch, maxlen);
	strncat(path, ".debug", maxlen);
}

/* registered_models_rwlock must be taken in write mode before calling this
 * function */
void _starpu_register_model(struct starpu_perfmodel_t *model)
{
	/* add the model to a linked list */
	struct starpu_model_list_t *node = (struct starpu_model_list_t *) malloc(sizeof(struct starpu_model_list_t));

	node->model = model;
	//model->debug_modelid = debug_modelid++;

	/* put this model at the beginning of the list */
	node->next = registered_models;
	registered_models = node;

#ifdef STARPU_MODEL_DEBUG
	_starpu_create_sampling_directory_if_needed();

	unsigned arch;
	unsigned nimpl;

	for (arch = 0; arch < STARPU_NARCH_VARIATIONS; arch++) {
		for (nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++) {
			char debugpath[256];
			starpu_perfmodel_debugfilepath(model, arch, debugpath, 256, nimpl);
			model->per_arch[arch][nimpl].debug_file = fopen(debugpath, "a+");
			STARPU_ASSERT(model->per_arch[arch][nimpl].debug_file);
		}
	}
#endif

	return;
}

static void get_model_path(struct starpu_perfmodel_t *model, char *path, size_t maxlen)
{
	_starpu_get_perf_model_dir_codelets(path, maxlen);
	strncat(path, model->symbol, maxlen);
	
	char hostname[32];
	char *forced_hostname = getenv("STARPU_HOSTNAME");
	if (forced_hostname && forced_hostname[0])
		snprintf(hostname, sizeof(hostname), "%s", forced_hostname);
	else
		gethostname(hostname, sizeof(hostname));
	strncat(path, ".", maxlen);
	strncat(path, hostname, maxlen);
}

static void save_history_based_model(struct starpu_perfmodel_t *model)
{
	STARPU_ASSERT(model);
	STARPU_ASSERT(model->symbol);

	/* TODO checks */

	/* filename = $STARPU_PERF_MODEL_DIR/codelets/symbol.hostname */
	char path[256];
	get_model_path(model, path, 256);

	_STARPU_DEBUG("Opening performance model file %s for model %s\n", path, model->symbol);

	/* overwrite existing file, or create it */
	FILE *f;
	f = fopen(path, "w+");
	STARPU_ASSERT(f);

	dump_model_file(f, model);

	fclose(f);
}

static void _starpu_dump_registered_models(void)
{
	PTHREAD_RWLOCK_WRLOCK(&registered_models_rwlock);

	struct starpu_model_list_t *node;
	node = registered_models;

	_STARPU_DEBUG("DUMP MODELS !\n");

	while (node) {
		save_history_based_model(node->model);		
		node = node->next;

		/* XXX free node */
	}

	PTHREAD_RWLOCK_UNLOCK(&registered_models_rwlock);
}

void _starpu_initialize_registered_performance_models(void)
{
	registered_models = NULL;

	PTHREAD_RWLOCK_INIT(&registered_models_rwlock, NULL);
}

void _starpu_deinitialize_registered_performance_models(void)
{
	if (_starpu_get_calibrate_flag())
		_starpu_dump_registered_models();

	PTHREAD_RWLOCK_DESTROY(&registered_models_rwlock);
}

/* We first try to grab the global lock in read mode to check whether the model
 * was loaded or not (this is very likely to have been already loaded). If the
 * model was not loaded yet, we take the lock in write mode, and if the model
 * is still not loaded once we have the lock, we do load it.  */
void _starpu_load_history_based_model(struct starpu_perfmodel_t *model, unsigned scan_history)
{

	STARPU_ASSERT(model);
	STARPU_ASSERT(model->symbol);
	
	int already_loaded;
 
	PTHREAD_RWLOCK_RDLOCK(&registered_models_rwlock);
	already_loaded = model->is_loaded;
	PTHREAD_RWLOCK_UNLOCK(&registered_models_rwlock);

	if (already_loaded)
		return;

	/* The model is still not loaded so we grab the lock in write mode, and
	 * if it's not loaded once we have the lock, we do load it. */

	PTHREAD_RWLOCK_WRLOCK(&registered_models_rwlock);

	/* Was the model initialized since the previous test ? */
	if (model->is_loaded)
	{
		PTHREAD_RWLOCK_UNLOCK(&registered_models_rwlock);
		return;
	}
	
	PTHREAD_RWLOCK_INIT(&model->model_rwlock, NULL);

	PTHREAD_RWLOCK_WRLOCK(&model->model_rwlock);

	/* make sure the performance model directory exists (or create it) */
	_starpu_create_sampling_directory_if_needed();

	/*
	 * We need to keep track of all the model that were opened so that we can 
	 * possibly update them at runtime termination ...
	 */
	_starpu_register_model(model);

	char path[256];
	get_model_path(model, path, 256);

	_STARPU_DEBUG("Opening performance model file %s for model %s ... ", path, model->symbol);

	unsigned calibrate_flag = _starpu_get_calibrate_flag();
	model->benchmarking = calibrate_flag; 
	
	/* try to open an existing file and load it */
	int res;
	res = access(path, F_OK); 
	if (res == 0) {
		if (calibrate_flag == 2)
		{
			/* The user specified that the performance model should
			 * be overwritten, so we don't load the existing file !
			 * */
                        _STARPU_DEBUG("Overwrite existing file\n");
			initialize_model(model);
		}
		else {
			/* We load the available file */
			_STARPU_DEBUG("File exists\n");
			FILE *f;
			f = fopen(path, "r");
			STARPU_ASSERT(f);
	
			parse_model_file(f, model, scan_history);
	
			fclose(f);
		}
	}
	else {
		_STARPU_DEBUG("File does not exists\n");
		if (!calibrate_flag) {
			_STARPU_DISP("Warning: model %s is not calibrated, forcing calibration for this run. Use the STARPU_CALIBRATE environment variable to control this.\n", model->symbol);
			_starpu_set_calibrate_flag(1);
			model->benchmarking = 1;
		}
		initialize_model(model);
	}

	model->is_loaded = 1;

	PTHREAD_RWLOCK_UNLOCK(&model->model_rwlock);

	PTHREAD_RWLOCK_UNLOCK(&registered_models_rwlock);
}

/* This function is intended to be used by external tools that should read
 * the performance model files */
int starpu_list_models(void)
{
        char path[256];
        DIR *dp;
        struct dirent *ep;

	char perf_model_dir_codelets[256];
	_starpu_get_perf_model_dir_codelets(perf_model_dir_codelets, 256);

        strncpy(path, perf_model_dir_codelets, 256);
        dp = opendir(path);
        if (dp != NULL) {
                while ((ep = readdir(dp))) {
                        if (strcmp(ep->d_name, ".") && strcmp(ep->d_name, ".."))
                                fprintf(stdout, "file: <%s>\n", ep->d_name);
                }
                closedir (dp);
                return 0;
        }
        else {
                perror ("Couldn't open the directory");
                return 1;
        }
}

/* This function is intended to be used by external tools that should read the
 * performance model files */
int starpu_load_history_debug(const char *symbol, struct starpu_perfmodel_t *model)
{
	model->symbol = strdup(symbol);

	/* where is the file if it exists ? */
	char path[256];
	get_model_path(model, path, 256);

	//	_STARPU_DEBUG("get_model_path -> %s\n", path);

	/* does it exist ? */
	int res;
	res = access(path, F_OK);
	if (res) {
		char *dot = strrchr(symbol, '.');
		if (dot) {
			char *symbol2 = strdup(symbol);
			symbol2[dot-symbol] = '\0';
			int ret;
			fprintf(stderr,"note: loading history from %s instead of %s\n", symbol2, symbol);
			ret = starpu_load_history_debug(symbol2,model);
			free(symbol2);
			return ret;
		}
		_STARPU_DISP("There is no performance model for symbol %s\n", symbol);
		return 1;
	}

	FILE *f = fopen(path, "r");
	STARPU_ASSERT(f);

	parse_model_file(f, model, 1);

	return 0;
}

void starpu_perfmodel_get_arch_name(enum starpu_perf_archtype arch, char *archname, size_t maxlen,unsigned nimpl)
{
	if (arch < STARPU_CUDA_DEFAULT)
	{
		if (arch == STARPU_CPU_DEFAULT)
		{
			/* NB: We could just use cpu_1 as well ... */
			snprintf(archname, maxlen, "cpu_impl_%u",nimpl);
		}
		else
		{
			/* For combined CPU workers */
			int cpu_count = arch - STARPU_CPU_DEFAULT + 1;
			snprintf(archname, maxlen, "cpu_%d_impl_%u", cpu_count,nimpl);
		}
	}
	else if ((STARPU_CUDA_DEFAULT <= arch)
		&& (arch < STARPU_CUDA_DEFAULT + STARPU_MAXCUDADEVS))
	{
		int devid = arch - STARPU_CUDA_DEFAULT;
		snprintf(archname, maxlen, "cuda_%d_impl_%u", devid,nimpl);
	}
	else if ((STARPU_OPENCL_DEFAULT <= arch)
		&& (arch < STARPU_OPENCL_DEFAULT + STARPU_MAXOPENCLDEVS))
	{
		int devid = arch - STARPU_OPENCL_DEFAULT;
		snprintf(archname, maxlen, "opencl_%d_impl_%u", devid,nimpl);
	}
	else if (arch == STARPU_GORDON_DEFAULT)
	{
		snprintf(archname, maxlen, "gordon_impl_%u",nimpl);
	}
	else
	{
		STARPU_ABORT();
	}
}

void starpu_perfmodel_debugfilepath(struct starpu_perfmodel_t *model,
		enum starpu_perf_archtype arch, char *path, size_t maxlen, unsigned nimpl)
{
	char archname[32];
	starpu_perfmodel_get_arch_name(arch, archname, 32, nimpl);

	STARPU_ASSERT(path);

	get_model_debug_path(model, archname, path, maxlen);
}

double _starpu_regression_based_job_expected_perf(struct starpu_perfmodel_t *model, enum starpu_perf_archtype arch, struct starpu_job_s *j, unsigned nimpl)
{
	double exp = -1.0;
	size_t size = _starpu_job_get_data_size(j);
	struct starpu_regression_model_t *regmodel;

	regmodel = &model->per_arch[arch][nimpl].regression;

	if (regmodel->valid)
                exp = regmodel->alpha*pow((double)size, regmodel->beta);

	return exp;
}

double _starpu_non_linear_regression_based_job_expected_perf(struct starpu_perfmodel_t *model, enum starpu_perf_archtype arch, struct starpu_job_s *j,unsigned nimpl)
{
	double exp = -1.0;
	size_t size = _starpu_job_get_data_size(j);
	struct starpu_regression_model_t *regmodel;

	regmodel = &model->per_arch[arch][nimpl].regression;

	if (regmodel->nl_valid)
		exp = regmodel->a*pow((double)size, regmodel->b) + regmodel->c;

	return exp;
}

double _starpu_history_based_job_expected_perf(struct starpu_perfmodel_t *model, enum starpu_perf_archtype arch, struct starpu_job_s *j,unsigned nimpl)
{
	double exp;
	struct starpu_per_arch_perfmodel_t *per_arch_model;
	struct starpu_history_entry_t *entry;
	struct starpu_htbl32_node_s *history;

	uint32_t key = _starpu_compute_buffers_footprint(j);

	per_arch_model = &model->per_arch[arch][nimpl];

	history = per_arch_model->history;
	if (!history)
		return -1.0;

	PTHREAD_RWLOCK_RDLOCK(&model->model_rwlock);
	entry = (struct starpu_history_entry_t *) _starpu_htbl_search_32(history, key);
	PTHREAD_RWLOCK_UNLOCK(&model->model_rwlock);

	exp = entry?entry->mean:-1.0;

	if (entry && entry->nsample < STARPU_CALIBRATION_MINIMUM)
		/* TODO: report differently if we've scheduled really enough
		 * of that task and the scheduler should perhaps put it aside */
		/* Not calibrated enough */
		exp = -1.0;

	if (exp == -1.0 && !model->benchmarking) {
		_STARPU_DISP("Warning: model %s is not calibrated enough, forcing calibration for this run. Use the STARPU_CALIBRATE environment variable to control this.\n", model->symbol);
		_starpu_set_calibrate_flag(1);
		model->benchmarking = 1;
	}

	return exp;
}

void _starpu_update_perfmodel_history(starpu_job_t j, struct starpu_perfmodel_t *model, enum starpu_perf_archtype arch, unsigned cpuid STARPU_ATTRIBUTE_UNUSED, double measured, unsigned nimpl)
{
	if (model)
	{
		PTHREAD_RWLOCK_WRLOCK(&model->model_rwlock);

		struct starpu_per_arch_perfmodel_t *per_arch_model = &model->per_arch[arch][nimpl];

		if (model->type == STARPU_HISTORY_BASED || model->type == STARPU_NL_REGRESSION_BASED)
		{
			uint32_t key = _starpu_compute_buffers_footprint(j);
			struct starpu_history_entry_t *entry;

			struct starpu_htbl32_node_s *history;
			struct starpu_htbl32_node_s **history_ptr;

			struct starpu_history_list_t **list;


			history = per_arch_model->history;
			history_ptr = &per_arch_model->history;
			list = &per_arch_model->list;

			entry = (struct starpu_history_entry_t *) _starpu_htbl_search_32(history, key);

			if (!entry)
			{
				/* this is the first entry with such a footprint */
				entry = (struct starpu_history_entry_t *) malloc(sizeof(struct starpu_history_entry_t));
				STARPU_ASSERT(entry);
					entry->mean = measured;
					entry->sum = measured;

					entry->deviation = 0.0;
					entry->sum2 = measured*measured;

					entry->size = _starpu_job_get_data_size(j);

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
		}
			
		if (model->type == STARPU_REGRESSION_BASED || model->type == STARPU_NL_REGRESSION_BASED)
		{
			struct starpu_regression_model_t *reg_model;
			reg_model = &per_arch_model->regression;

			/* update the regression model */
			size_t job_size = _starpu_job_get_data_size(j);
			double logy, logx;
			logx = log((double)job_size);
			logy = log(measured);

			reg_model->sumlnx += logx;
			reg_model->sumlnx2 += logx*logx;
			reg_model->sumlny += logy;
			reg_model->sumlnxlny += logx*logy;
			reg_model->nsample++;

			unsigned n = reg_model->nsample;
			
			double num = (n*reg_model->sumlnxlny - reg_model->sumlnx*reg_model->sumlny);
			double denom = (n*reg_model->sumlnx2 - reg_model->sumlnx*reg_model->sumlnx);

			reg_model->beta = num/denom;
			reg_model->alpha = exp((reg_model->sumlny - reg_model->beta*reg_model->sumlnx)/n);

			if (reg_model->nsample >= STARPU_CALIBRATION_MINIMUM) {
				reg_model->valid = 1;
				reg_model->nl_valid = 1;
			}
		}

#ifdef STARPU_MODEL_DEBUG
		struct starpu_task *task = j->task;
		FILE * debug_file = per_arch_model->debug_file;

		STARPU_ASSERT(j->footprint_is_computed);

		fprintf(debug_file, "0x%x\t%lu\t%f\t%f\t%f\t%d\t\t", j->footprint, (unsigned long) _starpu_job_get_data_size(j), measured, task->predicted, task->predicted_transfer, cpuid);
		unsigned i;
			
		for (i = 0; i < task->cl->nbuffers; i++)
		{
			starpu_data_handle handle = task->buffers[i].handle;

			STARPU_ASSERT(handle->ops);
			STARPU_ASSERT(handle->ops->display);
			handle->ops->display(handle, debug_file);
		}
		fprintf(debug_file, "\n");	

#endif
		
		PTHREAD_RWLOCK_UNLOCK(&model->model_rwlock);
	}
}
