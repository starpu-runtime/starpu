#include <unistd.h>
#include <core/perfmodel/perfmodel.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <datawizard/datawizard.h>

/*
 * PER ARCH model
 */

static double per_arch_job_expected_length(struct perfmodel_t *model, uint32_t who, struct job_s *j)
{
	double exp;
	
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

	return -1.0;
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

	return -1.0;
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

			case REGRESSION_BASED:
				return regression_based_job_expected_length(model, who, j);
				break;
			default:
				ASSERT(0);
		};
	}

	/* no model was found */
	return 0.0;
}
