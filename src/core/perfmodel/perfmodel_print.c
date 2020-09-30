/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu.h>
#include <starpu_perfmodel.h>
#include <common/config.h>
#include "perfmodel.h"

static
void _starpu_perfmodel_print_history_based(struct starpu_perfmodel_per_arch *per_arch_model, char *parameter, uint32_t *footprint, FILE *output)
{
	struct starpu_perfmodel_history_list *ptr;

	ptr = per_arch_model->list;

	if (!parameter && ptr)
		fprintf(output, "# hash\t\tsize\t\tflops\t\tmean (us)\tstddev (us)\t\tn\n");

	while (ptr)
	{
		struct starpu_perfmodel_history_entry *entry = ptr->entry;
		if (!footprint || entry->footprint == *footprint)
		{
			if (!parameter)
			{
				/* There isn't a parameter that is explicitely requested, so we display all parameters */
				fprintf(output, "%08x\t%-15lu\t%-15e\t%-15e\t%-15e\t%u\n", entry->footprint,
					(unsigned long) entry->size, entry->flops, entry->mean, entry->deviation, entry->nsample);
			}
			else
			{
				/* only display the parameter that was specifically requested */
				if (strcmp(parameter, "mean") == 0)
				{
					fprintf(output, "%-15e\n", entry->mean);
				}

				if (strcmp(parameter, "stddev") == 0)
				{
					fprintf(output, "%-15e\n", entry->deviation);
					return;
				}
			}
		}

		ptr = ptr->next;
	}
}

void starpu_perfmodel_print(struct starpu_perfmodel *model, struct starpu_perfmodel_arch* arch, unsigned nimpl, char *parameter, uint32_t *footprint, FILE *output)
{
	int comb = starpu_perfmodel_arch_comb_get(arch->ndevices, arch->devices);
	STARPU_ASSERT(comb != -1);

	struct starpu_perfmodel_per_arch *arch_model = &model->state->per_arch[comb][nimpl];

	if (arch_model->regression.nsample || arch_model->regression.valid || arch_model->regression.nl_valid || arch_model->list)
	{
		char archname[32];
		starpu_perfmodel_get_arch_name(arch, archname, 32, nimpl);
		fprintf(output, "# performance model for %s\n", archname);
	}

	if (parameter == NULL)
	{
		/* no specific parameter was requested, so we display everything */
		if (arch_model->regression.nsample)
		{
			fprintf(output, "\tRegression : #sample = %u\n", arch_model->regression.nsample);
		}

		/* Only display the regression model if we could actually build a model */
		if (arch_model->regression.valid)
		{
			fprintf(output, "\tLinear: y = alpha size ^ beta\n");
			fprintf(output, "\t\talpha = %e\n", arch_model->regression.alpha);
			fprintf(output, "\t\tbeta = %e\n", arch_model->regression.beta);
		}
		else
		{
			//fprintf(output, "\tLinear model is INVALID\n");
		}

		if (arch_model->regression.nl_valid)
		{
			fprintf(output, "\tNon-Linear: y = a size ^b + c\n");
			fprintf(output, "\t\ta = %e\n", arch_model->regression.a);
			fprintf(output, "\t\tb = %e\n", arch_model->regression.b);
			fprintf(output, "\t\tc = %e\n", arch_model->regression.c);
		}
		else
		{
			//fprintf(output, "\tNon-Linear model is INVALID\n");
		}

		_starpu_perfmodel_print_history_based(arch_model, parameter, footprint, output);

#if 0
		char debugname[1024];
		starpu_perfmodel_debugfilepath(model, arch, debugname, 1024, nimpl);
		_STARPU_MSG("\t debug file path : %s\n", debugname);
#endif
	}
	else
	{
		/* only display the parameter that was specifically requested */
		if (strcmp(parameter, "a") == 0)
		{
			fprintf(output, "%e\n", arch_model->regression.a);
			return;
		}

		if (strcmp(parameter, "b") == 0)
		{
			fprintf(output, "%e\n", arch_model->regression.b);
			return;
		}

		if (strcmp(parameter, "c") == 0)
		{
			fprintf(output, "%e\n", arch_model->regression.c);
			return;
		}

		if (strcmp(parameter, "alpha") == 0)
		{
			fprintf(output, "%e\n", arch_model->regression.alpha);
			return;
		}

		if (strcmp(parameter, "beta") == 0)
		{
			fprintf(output, "%e\n", arch_model->regression.beta);
			return;
		}

		if (strcmp(parameter, "path-file-debug") == 0)
		{
			char debugname[256];
			starpu_perfmodel_debugfilepath(model, arch, debugname, 256, nimpl);
			fprintf(output, "%s\n", debugname);
			return;
		}

		if ((strcmp(parameter, "mean") == 0) || (strcmp(parameter, "stddev") == 0))
		{
			_starpu_perfmodel_print_history_based(arch_model, parameter, footprint, output);
			return;
		}

		/* TODO display if it's valid ? */

		_STARPU_ERROR("Unknown parameter requested, aborting.\n");
	}
}

int starpu_perfmodel_print_all(struct starpu_perfmodel *model, char *arch, char *parameter, uint32_t *footprint, FILE *output)
{
	_starpu_init_and_load_perfmodel(model);
	if (arch == NULL)
	{
		int comb, impl;
		for(comb = 0; comb < starpu_perfmodel_get_narch_combs(); comb++)
		{
			struct starpu_perfmodel_arch *arch_comb = starpu_perfmodel_arch_comb_fetch(comb);
			int nimpls = model->state ? model->state->nimpls[comb] : 0;
			for(impl = 0; impl < nimpls; impl++)
				starpu_perfmodel_print(model, arch_comb, impl, parameter, footprint, output);
		}
	}
	else
	{
		if (strcmp(arch, "cpu") == 0)
		{
			int implid;
			struct starpu_perfmodel_arch perf_arch;
			perf_arch.ndevices = 1;
			_STARPU_MALLOC(perf_arch.devices, sizeof(struct starpu_perfmodel_device));
			perf_arch.devices[0].type = STARPU_CPU_WORKER;
			perf_arch.devices[0].devid = 0;
			perf_arch.devices[0].ncores = 1;
			int comb = starpu_perfmodel_arch_comb_get(perf_arch.ndevices, perf_arch.devices);
			STARPU_ASSERT(comb != -1);
			int nimpls = model->state->nimpls[comb];
			for (implid = 0; implid < nimpls; implid++)
				starpu_perfmodel_print(model, &perf_arch,implid, parameter, footprint, output); /* Display all codelets on cpu */
			free(perf_arch.devices);
			return 0;
		}

		int k;
		if (sscanf(arch, "cpu:%d", &k) == 1)
		{
			/* For combined CPU workers */
			if ((k < 1) || (k > STARPU_MAXCPUS))
			{
				_STARPU_ERROR("Invalid CPU size\n");
			}

			int implid;
			struct starpu_perfmodel_arch perf_arch;
			perf_arch.ndevices = 1;
			_STARPU_MALLOC(perf_arch.devices, sizeof(struct starpu_perfmodel_device));
			perf_arch.devices[0].type = STARPU_CPU_WORKER;
			perf_arch.devices[0].devid = 0;
			perf_arch.devices[0].ncores = k;
			int comb = starpu_perfmodel_arch_comb_get(perf_arch.ndevices, perf_arch.devices);
			STARPU_ASSERT(comb != -1);
			int nimpls = model->state->nimpls[comb];

			for (implid = 0; implid < nimpls; implid++)
				starpu_perfmodel_print(model, &perf_arch, implid, parameter, footprint, output);
			free(perf_arch.devices);
			return 0;
		}

		if (strcmp(arch, "cuda") == 0)
		{
			int implid;
			struct starpu_perfmodel_arch perf_arch;

			perf_arch.ndevices = 1;
			_STARPU_MALLOC(perf_arch.devices, sizeof(struct starpu_perfmodel_device));
			perf_arch.devices[0].type = STARPU_CUDA_WORKER;
			perf_arch.devices[0].ncores = 1;
			int comb;
			for(comb = 0; comb < starpu_perfmodel_get_narch_combs(); comb++)
			{
				struct starpu_perfmodel_arch *arch_comb = starpu_perfmodel_arch_comb_fetch(comb);
				if(arch_comb->ndevices == 1 && arch_comb->devices[0].type == STARPU_CUDA_WORKER)
				{
					perf_arch.devices[0].devid = arch_comb->devices[0].devid;
					int nimpls = model->state->nimpls[comb];

					for (implid = 0; implid < nimpls; implid++)
						starpu_perfmodel_print(model, &perf_arch, implid, parameter, footprint, output);
				}
			}
			free(perf_arch.devices);
			return 0;
		}

		/* TODO: There must be a cleaner way ! */
		int gpuid;
		int nmatched;
		nmatched = sscanf(arch, "cuda_%d", &gpuid);
		if (nmatched == 0)
			nmatched = sscanf(arch, "cuda%d", &gpuid);
		if (nmatched == 1)
		{
			struct starpu_perfmodel_arch perf_arch;
			perf_arch.ndevices = 1;
			_STARPU_MALLOC(perf_arch.devices, sizeof(struct starpu_perfmodel_device));

			perf_arch.devices[0].type = STARPU_CUDA_WORKER;
			perf_arch.devices[0].devid = gpuid;
			perf_arch.devices[0].ncores = 1;

			int comb = starpu_perfmodel_arch_comb_get(perf_arch.ndevices, perf_arch.devices);
			STARPU_ASSERT(comb != -1);
			int nimpls = model->state->nimpls[comb];

			int implid;
			for (implid = 0; implid < nimpls; implid++)
				starpu_perfmodel_print(model, &perf_arch, implid, parameter, footprint, output);
			return 0;
		}

		_STARPU_MSG("Unknown architecture requested\n");
		return -1;
	}
	return 0;
}

int starpu_perfmodel_print_estimations(struct starpu_perfmodel *model, uint32_t footprint, FILE *output)
{
	unsigned workerid;
	for (workerid = 0; workerid < starpu_worker_get_count(); workerid++)
	{
		struct starpu_perfmodel_arch* arch = starpu_worker_get_perf_archtype(workerid, STARPU_NMAX_SCHED_CTXS);
		int comb = starpu_perfmodel_arch_comb_get(arch->ndevices, arch->devices);
		struct starpu_perfmodel_per_arch *arch_model;
		struct starpu_perfmodel_history_list *ptr = NULL;

		if (comb >= 0 && model->state->per_arch[comb])
		{
			arch_model = &model->state->per_arch[comb][0];

			for (ptr = arch_model->list; ptr; ptr = ptr->next)
			{
				struct starpu_perfmodel_history_entry *entry = ptr->entry;
				if (entry->footprint == footprint)
				{
					fprintf(output, "%s%e", workerid?" ":"", entry->mean);
					break;
				}
			}
		}
		if (!ptr)
		{
			/* Didn't find any entry :/ */
			fprintf(output, "%sinf", workerid?" ":"");
		}
	}
	return 0;
}
