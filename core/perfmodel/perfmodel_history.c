#include <unistd.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <core/perfmodel/perfmodel.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <common/mutex.h>
#include <datawizard/datawizard.h>
#include <core/perfmodel/regression.h>

/*
 * History based model
 */


static void insert_history_entry(struct history_entry_t *entry, struct history_list_t **list, struct htbl32_node_s **history_ptr)
{
	struct history_list_t *link;
	struct history_entry_t *old;

	link = malloc(sizeof(struct history_list_t));
	link->next = *list;
	link->entry = entry;
	*list = link;

	old = htbl_insert_32(history_ptr, entry->footprint, entry);
	/* that may fail in case there is some concurrency issue */
	ASSERT(old == NULL);
}


static void dump_reg_model(FILE *f, struct regression_model_t *reg_model)
{
	fprintf(f, "%le\t%le\t%le\t%le\t%le\t%le\t%d\n", reg_model->sumlnx, reg_model->sumlnx2, reg_model->sumlny, reg_model->sumlnxlny, reg_model->alpha, reg_model->beta, reg_model->nsample);
}

static void scan_reg_model(FILE *f, struct regression_model_t *reg_model)
{
	int res;

	res = fscanf(f, "%le\t%le\t%le\t%le\t%le\t%le\t%d\n", &reg_model->sumlnx, &reg_model->sumlnx2, &reg_model->sumlny, &reg_model->sumlnxlny, &reg_model->alpha, &reg_model->beta, &reg_model->nsample);
	ASSERT(res == 7);
}


static void dump_history_entry(FILE *f, struct history_entry_t *entry)
{
	fprintf(f, "%x\t%zu\t%le\t%le\t%le\t%le\t%d\n", entry->footprint, entry->size, entry->mean, entry->deviation, entry->sum, entry->sum2, entry->nsample);
}

static void scan_history_entry(FILE *f, struct history_entry_t *entry)
{
	int res;

	res = fscanf(f, "%x\t%zu\t%le\t%le\t%le\t%le\t%d\n", &entry->footprint, &entry->size, &entry->mean, &entry->deviation, &entry->sum, &entry->sum2, &entry->nsample);
	ASSERT(res == 7);
}

static void parse_model_file(FILE *f, struct perfmodel_t *model)
{
	/* header */
	unsigned ncore_entries, ncuda_entries;
	int res = fscanf(f, "%d\n%d\n", &ncore_entries, &ncuda_entries);
	ASSERT(res == 2);

	/* parse regression models */
	scan_reg_model(f, &model->regression_core);
	scan_reg_model(f, &model->regression_cuda);

	/* for now, we just "consume" that */
	double a, b, c;
	res = fscanf(f, "%le\t%le\t%le\n", &a, &b, &c);
	ASSERT(res == 3);
	res = fscanf(f, "%le\t%le\t%le\n", &a, &b, &c);
	ASSERT(res == 3);

	/* parse core entries */
	unsigned i;
	for (i = 0; i < ncore_entries; i++) {
		struct history_entry_t *entry = malloc(sizeof(struct history_entry_t));
		ASSERT(entry);

		scan_history_entry(f, entry);
		
		/* insert the entry in the hashtable and the list structures  */
		insert_history_entry(entry, &model->list_core, &model->history_core);
	}

	for (i = 0; i < ncuda_entries; i++) {
		struct history_entry_t *entry = malloc(sizeof(struct history_entry_t));
		ASSERT(entry);
		
		scan_history_entry(f, entry);

		/* insert the entry in the hashtable and the list structures  */
		insert_history_entry(entry, &model->list_cuda, &model->history_cuda);
	}
	
//	model->benchmarking = 0;
}


static void dump_model_file(FILE *f, struct perfmodel_t *model)
{
	/* count the number of elements in the lists */
	struct history_list_t *ptr;

	unsigned ncore_entries = 0;
	ptr = model->list_core;
	while(ptr) {
		ncore_entries++;
		ptr = ptr->next;
	}

	unsigned ncuda_entries = 0;
	ptr = model->list_cuda;
	while(ptr) {
		ncuda_entries++;
		ptr = ptr->next;
	}

	/* header */
	fprintf(f, "%d\n", ncore_entries);
	fprintf(f, "%d\n", ncuda_entries);

	dump_reg_model(f, &model->regression_core);
	dump_reg_model(f, &model->regression_cuda);

	/* TODO clean up !*/
	double a,b,c;
	regression_non_linear_power(model->list_core, &a, &b, &c);
	fprintf(f, "%le\t%le\t%le\n", a, b, c);
	regression_non_linear_power(model->list_cuda, &a, &b, &c);
	fprintf(f, "%le\t%le\t%le\n", a, b, c);

	ptr = model->list_core;
	while (ptr) {
		//memcpy(&entries_array[i++], ptr->entry, sizeof(struct history_entry_t));
		dump_history_entry(f, ptr->entry);
		ptr = ptr->next;
	}

	ptr = model->list_cuda;
	while (ptr) {
		//memcpy(&entries_array[i++], ptr->entry, sizeof(struct history_entry_t));
		dump_history_entry(f, ptr->entry);
		ptr = ptr->next;
	}

}

static void initialize_model(struct perfmodel_t *model)
{
	model->history_core = NULL;
	model->history_cuda = NULL;

	model->list_core = NULL;
	model->list_cuda = NULL;

//	model->benchmarking = 1;
}

static struct model_list_t *registered_models = NULL;
//static unsigned debug_modelid = 0;

static void get_model_debug_path(struct perfmodel_t *model, const char *arch, char *path, size_t maxlen)
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


void register_model(struct perfmodel_t *model)
{
	/* add the model to a linked list */
	struct model_list_t *node = malloc(sizeof(struct model_list_t));

	node->model = model;
	//model->debug_modelid = debug_modelid++;

	/* put this model at the beginning of the list */
	node->next = registered_models;
	registered_models = node;

#ifdef MODEL_DEBUG
	char debugpath[256];
	get_model_debug_path(model, "cuda", debugpath, 256);
	model->cuda_debug_file = fopen(debugpath, "a+");
	ASSERT(model->cuda_debug_file);

	get_model_debug_path(model, "core", debugpath, 256);
	model->core_debug_file = fopen(debugpath, "a+");
	ASSERT(model->core_debug_file);
#endif

	return;
}

static void get_model_path(struct perfmodel_t *model, char *path, size_t maxlen)
{
	strncpy(path, PERF_MODEL_DIR, maxlen);
	strncat(path, model->symbol, maxlen);
	
	char hostname[32];
	gethostname(hostname, 32);
	strncat(path, ".", maxlen);
	strncat(path, hostname, maxlen);
}

void save_history_based_model(struct perfmodel_t *model)
{
	ASSERT(model);
	ASSERT(model->symbol);

	/* TODO checks */

	/* filename = $PERF_MODEL_DIR/symbol.hostname */
	char path[256];
	get_model_path(model, path, 256);

	fprintf(stderr, "Opening performance model file %s for model %s\n", path, model->symbol);

	/* overwrite existing file, or create it */
	FILE *f;
	f = fopen(path, "w+");
	ASSERT(f);

	dump_model_file(f, model);

	fclose(f);

#ifdef DEBUG_MODEL
	fclose(model->cuda_debug_file);
	fclose(model->core_debug_file);
#endif
}

void dump_registered_models(void)
{
	struct model_list_t *node;
	node = registered_models;

	fprintf(stderr, "DUMP MODELS !\n");

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
		ASSERT(errno == EEXIST);

		/* make sure that it is actually a directory */
		struct stat sb;
		stat(PERF_MODEL_DIR, &sb);
		ASSERT(S_ISDIR(sb.st_mode));
	}
}

void load_history_based_model(struct perfmodel_t *model)
{
	ASSERT(model);
	ASSERT(model->symbol);

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
	
		fprintf(stderr, "Opening performance model file %s for model %s\n", path, model->symbol);
	
		/* try to open an existing file and load it */
		int res;
		res = access(path, F_OK); 
		if (res == 0) {
			fprintf(stderr, "File exists !\n");
	
			FILE *f;
			f = fopen(path, "r");
			ASSERT(f);
	
			parse_model_file(f, model);
	
			fclose(f);
		}
		else {
			//fprintf(stderr, "File does not exists !\n");
			initialize_model(model);
		}
	
	
		if (get_env_number("CALIBRATE") != -1)
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

double history_based_job_expected_length(struct perfmodel_t *model, uint32_t who, struct job_s *j)
{
	double exp;

	if (!model->is_loaded)
		load_history_based_model(model);

	if (!j->footprint_is_computed)
		compute_buffers_footprint(j);
		
	uint32_t key = j->footprint;
	struct history_entry_t *entry;

	struct htbl32_node_s *history;
	struct htbl32_node_s **history_ptr;
	struct history_list_t **list;

	if ( who & (CUBLAS|CUDA)) {
		history = model->history_cuda;
		history_ptr = &model->history_cuda;
		list = &model->list_cuda;
	}
	else if ( who & CORE) {
		history = model->history_core;
		history_ptr = &model->history_core;
		list = &model->list_core;
	}
	else {
		/* XXX cleanup */
		ASSERT(0);
	}

	take_mutex(&model->model_mutex);
	entry = htbl_search_32(history, key);
	release_mutex(&model->model_mutex);

	exp = entry?entry->mean:-1.0;

//	fprintf(stderr, "history prediction : entry = %p (footprint %x), expected %e\n", entry, j->footprint, exp);

	return exp;
}

void update_perfmodel_history(job_t j, enum archtype arch, double measured)
{
	if (j->model)
	{
		uint32_t key = j->footprint;
		struct history_entry_t *entry;

		struct htbl32_node_s *history;
		struct htbl32_node_s **history_ptr;
		struct regression_model_t *reg_model;

		struct history_list_t **list;

		ASSERT(j->model);

		switch (arch) {
			case CORE_WORKER:
				history = j->model->history_core;
				history_ptr = &j->model->history_core;
				reg_model = &j->model->regression_core;
				list = &j->model->list_core;
				break;
			case CUDA_WORKER:
				history = j->model->history_cuda;
				history_ptr = &j->model->history_cuda;
				reg_model = &j->model->regression_cuda;
				list = &j->model->list_cuda;
				break;
			default:
				ASSERT(0);
		}

		take_mutex(&j->model->model_mutex);

		entry = htbl_search_32(history, key);

		if (!entry)
		{
			/* this is the first entry with such a footprint */
			entry = malloc(sizeof(struct history_entry_t));
			ASSERT(entry);
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
		
		release_mutex(&j->model->model_mutex);

		ASSERT(entry);

#ifdef MODEL_DEBUG
		FILE * debug_file = (arch == CUDA_WORKER) ? j->model->cuda_debug_file:j->model->core_debug_file;

		fprintf(debug_file, "%lf\t", measured);
		unsigned i;
		for (i = 0; i < j->nbuffers; i++)
		{
			data_state *state = j->buffers[i].state;

			ASSERT(state->ops);
			ASSERT(state->ops->display);
			
			state->ops->display(state, debug_file);
		}
		fprintf(debug_file, "\n");	
#endif
	}
}
