#include <unistd.h>
#include <core/perfmodel/perfmodel.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <datawizard/datawizard.h>

/*
 * PER ARCH model
 */

static double per_arch_job_expected_length(struct perfmodel_t *model, enum perf_archtype arch, struct job_s *j)
{
	double exp = -1.0;
	double (*per_arch_cost_model)(struct buffer_descr_t *);
	
	if (!model->is_loaded)
	{
		if (get_env_number("CALIBRATE") != -1)
		{
			fprintf(stderr, "CALIBRATE model %s\n", model->symbol);
			model->benchmarking = 1;
		}
		else {
			model->benchmarking = 0;
		}
		
		register_model(model);
		model->is_loaded = 1;
	}

	per_arch_cost_model = model->per_arch[arch].cost_model;

	if (per_arch_cost_model)
		exp = per_arch_cost_model(j->buffers);

	return exp;
}

/*
 * Common model
 */

static double common_job_expected_length(struct perfmodel_t *model, uint32_t who, struct job_s *j)
{
	double exp;

	if (model->cost_model) {
		float alpha;
		exp = model->cost_model(j->buffers);
		switch (who) {
			case CORE:
				alpha = CORE_ALPHA;
				break;
			case CUDA:
				alpha = CUDA_ALPHA;
				break;
			default:
				/* perhaps there are various worker types on that queue */
				alpha = 1.0; // this value is not significant ...
				break;
		}

		ASSERT(alpha != 0.0f);

		return (exp/alpha);
	}

	return -1.0;
}

double job_expected_length(uint32_t who, struct job_s *j, enum perf_archtype arch)
{
	struct perfmodel_t *model = j->model;

	if (model) {
		switch (model->type) {
			case PER_ARCH:
				return per_arch_job_expected_length(model, arch, j);

			case COMMON:
				return common_job_expected_length(model, who, j);

			case HISTORY_BASED:
				return history_based_job_expected_length(model, arch, j);

			case REGRESSION_BASED:
				return regression_based_job_expected_length(model, arch, j);

			default:
				ASSERT(0);
		};
	}

	/* no model was found */
	return 0.0;
}
