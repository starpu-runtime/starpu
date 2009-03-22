#include <stdio.h>
#include <stdint.h>

#include "fxt-tool.h"

static char *out_path = "dag.dot";
static FILE *out_file;


void init_dag_dot(void)
{
	/* create a new file */
	out_file = fopen(out_path, "w+");

	fprintf(out_file, "digraph G {\n");
	fprintf(out_file, "\tcolor=white\n");
	fprintf(out_file, "\trankdir=LR;\n");
}

void terminate_dat_dot(void)
{
	fprintf(out_file, "}\n");
	fclose(out_file);
}

void add_deps(uint64_t child, uint64_t father)
{
	fprintf(out_file, "\t \"%llx\"->\"%llx\"\n", 
		(unsigned long long)father, (unsigned long long)child);
}

void dot_set_tag_done(uint64_t tag, char *color)
{

	fprintf(out_file, "\t \"%llx\" \[ style=filled, label=\"\", color=\"%s\"]\n", 
		(unsigned long long)tag, color);
}
