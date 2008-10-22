#include <unistd.h>
#include <core/perfmodel.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <datawizard/footprint.h>

//#define PER_ARCH_MODEL	1

/*
 * PER ARCH model
 */

static double per_arch_job_expected_length(struct perfmodel_t *model, uint32_t who, struct job_s *j)
{
	double exp;

	if ( (who & (CUBLAS|CUDA)) && model->cuda_cost_model) {
		/* use CUDA model */
		#ifdef TRANSFER_OVERHEAD
		exp = model->cuda_cost_model(j->buffers)*1.15;
		#else
		exp = model->cuda_cost_model(j->buffers) + 0.0;
		#endif
		return exp;
	}

	if ( (who & CORE) && model->core_cost_model) {
		/* use CORE model */
		exp = model->core_cost_model(j->buffers);
		return exp;
	}

	return 0.0;
}

/*
 * Common model
 */

static double common_job_expected_length(struct perfmodel_t *model, uint32_t who, struct job_s *j)
{
	double exp;

	if (model->cost_model) {
		/* XXX fix ! */
		exp = 0.0;
		return exp;
	}

	return 0.0;
}

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


static void parse_model_file(FILE *f, struct perfmodel_t *model)
{
	/* TODO */
	
	/* header */
	unsigned ncore_entries, ncuda_entries;
	int res = fscanf(f, "%d\n%d\n", &ncore_entries, &ncuda_entries);
	ASSERT(res == 2);

	/* parse core entries */
	unsigned i;
	for (i = 0; i < ncore_entries; i++) {
		struct history_entry_t *entry = malloc(sizeof(struct history_entry_t));
		ASSERT(entry);
		res = fscanf(f, "%x\t%lf\t%d\n", &entry->footprint, &entry->measured, &entry->nsample);
		ASSERT(res == 3);
		
		/* insert the entry in the hashtable and the list structures  */
		insert_history_entry(entry, &model->list_core, &model->history_core);
	}

	for (i = 0; i < ncuda_entries; i++) {
		struct history_entry_t *entry = malloc(sizeof(struct history_entry_t));
		ASSERT(entry);
		res = fscanf(f, "%x\t%lf\t%d\n", &entry->footprint, &entry->measured, &entry->nsample);
		ASSERT(res == 3);
		
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

	unsigned i = 0;

	/* header */
	fprintf(f, "%d\n", ncore_entries);
	fprintf(f, "%d\n", ncuda_entries);

	ptr = model->list_core;
	while (ptr) {
		//memcpy(&entries_array[i++], ptr->entry, sizeof(struct history_entry_t));
		fprintf(f, "%x\t%lf\t%d\n", ptr->entry->footprint, ptr->entry->measured, ptr->entry->nsample);
		ptr = ptr->next;
	}

	ptr = model->list_cuda;
	while (ptr) {
		//memcpy(&entries_array[i++], ptr->entry, sizeof(struct history_entry_t));
		fprintf(f, "%x\t%lf\t%d\n", ptr->entry->footprint, ptr->entry->measured, ptr->entry->nsample);
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

void register_model(struct perfmodel_t *model)
{
	/* add the model to a linked list */
	struct model_list_t *node = malloc(sizeof(struct model_list_t));

	node->model = model;
	node->next = registered_models;
	registered_models = node;

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

void load_history_based_model(struct perfmodel_t *model)
{
	ASSERT(model);
	ASSERT(model->symbol);

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

static double history_based_job_expected_length(struct perfmodel_t *model, uint32_t who, struct job_s *j)
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

	entry = htbl_search_32(history, key);

	exp = entry?entry->measured:0.0;

//	fprintf(stderr, "history prediction : entry = %p (footprint %x), expected %e\n", entry, j->footprint, exp);

	return exp;
}

double job_expected_length(uint32_t who, struct job_s *j)
{
	double exp;
	struct perfmodel_t *model = j->model;

	if (model) {
		switch (model->type) {
			case PER_ARCH:
				return per_arch_job_expected_length(model, who, j);
				break;

			case COMMON:
				return common_job_expected_length(model, who, j);
				break;

			case HISTORY_BASED:
				return history_based_job_expected_length(model, who, j);
				break;
			default:
				ASSERT(0);
		};
	}

	/* no model was found */
	return 0.0;
}


void update_perfmodel_history(job_t j, enum archtype arch, double measured)
{
	if (j->model)
	{
		uint32_t key = j->footprint;
		struct history_entry_t *entry;

		struct htbl32_node_s *history;
		struct htbl32_node_s **history_ptr;

		struct history_list_t **list;

		ASSERT(j->model);

		switch (arch) {
			case CORE_WORKER:
				history = j->model->history_core;
				history_ptr = &j->model->history_core;
				list = &j->model->list_core;
				break;
			case CUDA_WORKER:
				history = j->model->history_cuda;
				history_ptr = &j->model->history_cuda;
				list = &j->model->list_cuda;
				break;
			default:
				ASSERT(0);
		}

		entry = htbl_search_32(history, key);

		if (!entry)
		{
			/* this is the first entry with such a footprint */
			entry = malloc(sizeof(struct history_entry_t));
			ASSERT(entry);
				entry->measured = measured;
				entry->footprint = key;
				entry->nsample = 1;

			insert_history_entry(entry, list, history_ptr);

		}
		else {
			/* there is already some entry with the same footprint */
			double oldmean = entry->measured;
			entry->measured =
				(oldmean * entry->nsample + measured)/(entry->nsample+1);
			entry->nsample++;
		}

		ASSERT(entry);

#ifdef MODEL_DEBUG
		fprintf(stderr, "model was %e, got %e (mean %e, footprint %x) factor (%2.2f \%%)\n",
				j->predicted, measured, entry->measured, key, 100*(measured/j->predicted - 1.0f));
#endif
	}
}
