/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2011       Télécom-SudParis
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
#include <config.h>
#include <assert.h>
#include <getopt.h>
#include <unistd.h>
#include <stdio.h>

#include <starpu.h>
#include <common/utils.h>
#include <common/uthash.h>
#include <core/perfmodel/perfmodel.h> // we need to browse the list associated to history-based models
                                      // just like in starpu_perfmodel_plot

#define STRHEADCMP(s, head) strncmp(s, head, strlen(head))


#if defined(_WIN32) && !defined(__CYGWIN__)
#include <windows.h>
#endif

#define PROGNAME "starpu_perfmodel_recdump"

struct _footprint_list
{
	struct _footprint_list* next;
	uint32_t footprint;
};

struct _footprint_list* add_footprint(struct _footprint_list* list, uint32_t footprint)
{
	struct _footprint_list * l = list;
	while(l)
	{
		if(l->footprint == footprint) break;
		l = l->next;
	}
	if(l) return list;
	else
	{
		struct _footprint_list * res = malloc(sizeof(struct _footprint_list));
		res->footprint = footprint;
		res->next = list;
		return res;
	}
}

static struct model
{
	UT_hash_handle hh;
	char *name;
	struct starpu_perfmodel model;
	struct _footprint_list* footprints;
} *models;

void get_comb_name(int comb, char* name, int name_size)
{
	struct starpu_perfmodel_arch *arch_comb = starpu_perfmodel_arch_comb_fetch(comb);
	STARPU_ASSERT_MSG(arch_comb->ndevices == 1, "Cannot work with multi-device workers\n");
	snprintf(name, name_size, "%s%d", starpu_perfmodel_get_archtype_name(arch_comb->devices[0].type), arch_comb->devices[0].devid);
}

void print_archs(FILE* output)
{
	int nb_workers = 0;
	unsigned workerid, node; int comb, old_comb = -1;

	fprintf(output, "%%rec: worker_count\n\n");
	for (workerid = 0; workerid < starpu_worker_get_count(); workerid++)
	{
		struct starpu_perfmodel_arch* arch = starpu_worker_get_perf_archtype(workerid, STARPU_NMAX_SCHED_CTXS);
		comb = starpu_perfmodel_arch_comb_get(arch->ndevices, arch->devices);
		if (comb < 0) continue; // Ignore architecture which is not present in any perfmodel

		if(comb != old_comb)
		{
			if(nb_workers > 0)
			{
				char name[32];
				get_comb_name(old_comb, name, 32);
				fprintf(output, "Architecture: %s\n", name);
				fprintf(output, "NbWorkers: %d\n\n", nb_workers);
			}
			old_comb = comb;
			nb_workers = 1;
		}
		else
		{
			nb_workers += 1;
		}
	}

	if(nb_workers > 0)
	{
		char name[32];
		get_comb_name(old_comb, name, 32);
		fprintf(output, "Architecture: %s\n", name);
		fprintf(output, "NbWorkers: %d\n\n", nb_workers);
	}

	fprintf(output, "%%rec: memory_workers\n\n");
	for (node = 0; node < starpu_memory_nodes_get_count(); node++)
	{
		unsigned printed = 0;
		char name[32];
		fprintf(output, "MemoryNode: %u\n", node);
		starpu_memory_node_get_name(node, name, sizeof(name));
		fprintf(output, "Name: %s\n", name);
		fprintf(output, "Size: %ld\n", (long) starpu_memory_get_total(node));
		for (workerid = 0; workerid < starpu_worker_get_count(); workerid++)
		{
			if (starpu_worker_get_memory_node(workerid) == node)
			{
				if (!printed)
				{
					fprintf(output, "Workers:");
					printed = 1;
				}
				fprintf(output, " %u", workerid);
			}
		}
		if (printed)
			fprintf(output, "\n");
		fprintf(output, "\n");
	}
}

/* output file name */
static char* poutput = NULL;
static char* pinput = NULL;

static void usage()
{
	fprintf(stderr, "Dumps perfmodels to a rec file\n\n");
	fprintf(stderr, "Usage: %s [ input-file ] [ -o output-file ]\n", PROGNAME);
        fprintf(stderr, "\n");
	fprintf(stderr, "If input or output file names are not given, stdin and stdout are used.");
	fprintf(stderr, "\n");
        fprintf(stderr, "Report bugs to <"PACKAGE_BUGREPORT">.");
        fprintf(stderr, "\n");
}

static void print_entry(const char *name, const char *archname, FILE *output, struct starpu_perfmodel_history_entry *entry)
{
	fprintf(output, "Model: %s\n", name);
	fprintf(output, "Architecture: %s\n", archname);
	fprintf(output, "Footprint: %08x\n", entry->footprint);
	fprintf(output, "Size: %lu\n", (unsigned long) entry->size);
	if (!isnan(entry->flops))
		fprintf(output, "Flops: %-15e\n", entry->flops);
	fprintf(output, "Mean: %-15e\nStddev: %-15e\n",
		entry->mean, entry->deviation);
	fprintf(output, "Samples: %u\n", entry->nsample);
	fprintf(output, "\n");
}

static void parse_args(int argc, char **argv)
{
	int c;

	static struct option long_options[] =
	{
		{"help",      no_argument, NULL, 'h'},
		{"output",     required_argument, NULL, 'o'},
		{0, 0, 0, 0}
	};

	int option_index;
	while ((c = getopt_long(argc, argv, "ho:", long_options, &option_index)) != -1)
	{
		switch (c)
		{
                case 'h': /* display help */
		  usage();
		  exit(EXIT_SUCCESS);
		  break;

		case 'o':
		  poutput = optarg;
		  break;
		case '?':
		default:
			fprintf(stderr, "Unrecognized option: -%c\n", optopt);
		}
	}

	if(optind < argc)
	{
		pinput = argv[optind++];
		if(optind < argc)
		{
			fprintf(stderr, "Unrecognized argument: %s\n", argv[optind]);
			exit(EXIT_FAILURE);
		}
	}
}

int main(int argc, char **argv)
{
#if defined(_WIN32) && !defined(__CYGWIN__) && !defined(__MINGW32__)
	WSADATA wsadata;
	WSAStartup(MAKEWORD(1,0), &wsadata);
	_STARPU_MSG("Listing perfmodels is not implemented on pure Windows yet\n");
	return 1;
#else
	FILE* output;
	parse_args(argc, argv);

	if(poutput != NULL)
	{
		output = fopen(poutput, "w+");
		if (!output)
		{
			fprintf(stderr, "couldn't open %s for write: %s\n", poutput, strerror(errno));
			exit(EXIT_FAILURE);
		}
	}
	else
	{
		output = stdout;
	}

	if (starpu_init(NULL) != 0)
	{
		fprintf(stderr, "StarPU initialization failure\n");
		exit(EXIT_FAILURE);
	}
	starpu_pause();

	if(pinput)
	{
		FILE* input = fopen(pinput, "r");
		char s[1024], *c;
		struct model *model, *tmp=NULL;
		uint32_t footprint = 0;
		char *model_name = NULL;
		int ret;

		if (!input)
		{
			fprintf(stderr, "couldn't open %s for read: %s\n", pinput, strerror(errno));
			exit(EXIT_FAILURE);
		}

		while (fgets(s, sizeof(s), input))
		{
			if (strlen(s) == sizeof(s) - 1)
			{
				fprintf(stderr, "oops, very long line '%s', it's odd\n", s);
				exit(EXIT_FAILURE);
			}

			if (s[0] == '\n')
			{
				/* empty line, end of task */
				if (model_name)
				{
					/* Try to get already-loaded model */
					HASH_FIND_STR(models, model_name, model);
					if (model == NULL)
					{
						model = malloc(sizeof(*model));
						model->name = model_name;
						model->footprints = NULL;
						memset(&model->model, 0, sizeof(model->model));
						model->model.type = STARPU_PERFMODEL_INVALID;
						ret = starpu_perfmodel_load_symbol(model_name, &model->model);
						if (ret == 1)
						{
							fprintf(stderr, "The performance model for the symbol <%s> could not be loaded\n", model_name);
							exit(EXIT_FAILURE);
						}
						HASH_ADD_STR(models, name, model);
					}
					else
					{
						free(model_name);
					}
					model->footprints = add_footprint(model->footprints, footprint);
					model_name = NULL;
				}
				continue;
			}

			/* Get rec field name */
			c = strchr(s, ':');
			if (!c)
			{
				fprintf(stderr, "odd line '%s'\n", s);
				exit(EXIT_FAILURE);
			}

			if (!STRHEADCMP(s, "Footprint: "))
			{
				footprint = strtoul(s + strlen("Footprint: "), NULL, 16);
			}
			else if (!STRHEADCMP(s, "Model: "))
			{
				model_name = strdup(s + strlen("Model: "));
				model_name[strlen(model_name) - 1] = '\0'; /* Drop '\n' */
			}
		}

		/* All models loaded */
		{
			print_archs(output);

			fprintf(output, "%%rec: timing\n\n");

			int nb_combs = starpu_perfmodel_get_narch_combs();

			HASH_ITER(hh, models, model, tmp)
			{
				struct _footprint_list* l = model->footprints, *ltmp;
				int comb;
				while(l)
				{
					for(comb = 0; comb < nb_combs; comb++)
					{
						char archname[32];
						get_comb_name(comb, archname, 32);

						if(!model->model.state || model->model.state->nimpls[comb] == 0)
						{
							_STARPU_DISP("Symbol %s does not have any implementation on comb %d, not dumping\n", model->name, comb);
							continue;
						}

						if(model->model.state->nimpls[comb] > 1)
							_STARPU_DISP("Warning, more than one implementations in comb %d of symbol %s, using only the first one\n", comb, model->name);

						struct starpu_perfmodel_per_arch *arch_model = &model->model.state->per_arch[comb][0];
						struct starpu_perfmodel_history_list *ptr;

						ptr = arch_model->list;
						if(!ptr)
							_STARPU_DISP("Implementation %d of symbol %s does not have history based model, not dumping\n", comb,  model->name);
						else while(ptr)
						     {
							     struct starpu_perfmodel_history_entry *entry = ptr->entry;
							     if(entry->footprint == l->footprint)
							     {
								     print_entry(model->name, archname, output, entry);
								     break;
							     }
							     ptr=ptr->next;
						     }
					}
					ltmp = l->next;
					free(l);
					l = ltmp;
				}

				free(model->name);
				HASH_DEL(models, model);
			}
		}
		fclose(input);
	}
	else
	{
		fprintf(output, "%%rec: timing\n\n");

		char *path;
		DIR *dp;
		struct dirent *ep;

		path = _starpu_get_perf_model_dir_codelet();
		dp = opendir(path);
		if (dp != NULL)
		{
			while ((ep = readdir(dp)))
			{
				if (strcmp(ep->d_name, ".") && strcmp(ep->d_name, ".."))
				{
					int comb, nb_combs;
					char* symbol = strdup(ep->d_name);
					char *dot = strrchr(symbol, '.');
					struct starpu_perfmodel model = {.type = STARPU_PERFMODEL_INVALID };

					if(dot) *dot = '\0';
					if (starpu_perfmodel_load_symbol(symbol, &model) != 0)
					{
						free(symbol);
						continue;
					}
					if(model.state == NULL)
					{
						free(symbol);
						continue;
					}

					_STARPU_DISP("Dumping %s\n", symbol);

					nb_combs = starpu_perfmodel_get_narch_combs();
					for(comb = 0; comb < nb_combs; ++comb)
					{
						char name[32];
						get_comb_name(comb, name, 32);

						if(!model.state || model.state->nimpls[comb] == 0)
						{
							_STARPU_DISP("Symbol %s does not have any implementation on comb %d, not dumping\n", symbol, comb);
							fprintf(output, "\n");
							continue;
						}

						struct starpu_perfmodel_per_arch *arch_model = &model.state->per_arch[comb][0];
						struct starpu_perfmodel_history_list *ptr;

						ptr = arch_model->list;
						if(!ptr)
							_STARPU_DISP("Symbol %s for comb %d does not have history based model, not dumping\n", symbol,  comb);
						else while(ptr)
						     {
							     print_entry(symbol, name, output, ptr->entry);
							     ptr=ptr->next;
						     }
					}
					starpu_perfmodel_unload_model(&model);
					free(symbol);
				}
			}
			closedir (dp);
		}
		else
		{
			_STARPU_DISP("Could not open the perfmodel directory <%s>: %s\n", path, strerror(errno));
		}

		print_archs(output);
	}

	starpu_resume();
	starpu_shutdown();

	fclose(output);
	return 0;
#endif
}
