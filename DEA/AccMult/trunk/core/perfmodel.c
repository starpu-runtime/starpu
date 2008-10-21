#include <core/perfmodel.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <datawizard/footprint.h>

//#define PER_ARCH_MODEL	1

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

void load_history_based_model(struct perfmodel_t *model)
{
	/* TODO */
}

void save_history_based_model(struct perfmodel_t *model)
{
	/* TODO */
}

static double history_based_job_expected_length(struct perfmodel_t *model, uint32_t who, struct job_s *j)
{
	double exp;

	if (!model->is_loaded)
	{
		/* load performance file */
		load_history_based_model(model);

		model->is_loaded = 1;
	}

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

	fprintf(stderr, "history prediction : entry = %p (footprint %x), expected %e\n", entry, j->footprint, exp);

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
			struct history_entry_t *old;

			entry = malloc(sizeof(struct history_entry_t));
			ASSERT(entry);
				entry->measured = measured;
				entry->footprint = key;
				entry->nsample = 1;
			
			struct history_list_t *link;
				link = malloc(sizeof(struct history_list_t));
				link->next = *list;
				link->entry = entry;
				*list = link;

			old = htbl_insert_32(history_ptr, key, entry);
			/* that may fail in case there is some concurrency issue */
			ASSERT(old == NULL);

		}
		else {
			/* there is already some entry with the same footprint */
			double oldmean = entry->measured;
			entry->measured =
				(oldmean * entry->nsample + measured)/(entry->nsample+1);
			entry->nsample++;
		}

		ASSERT(entry);

		fprintf(stderr, "model was %e, got %e (mean %e, footprint %x) factor (%2.2f \%%)\n",
				j->predicted, measured, entry->measured, key, 100*(measured/j->predicted - 1.0f));

	}
}
