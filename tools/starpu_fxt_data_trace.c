#include <stdio.h>
#include <config.h>
#include <starpu.h>
#include <string.h>

#define PROGNAME "starpu_fxt_data_trace"
#define MAX_LINE_SIZE 100

static void usage(char *progname)
{
	fprintf(stderr, "Usage : %s <filename> [codelet...]\n", progname);
	exit(77);
}

static void write_plt(int argc, char **argv){
	FILE *codelet_list = fopen("codelet_list", "r");
	if(!codelet_list)
	{
		perror("Error while opening codelet list:");
		exit(-1);
	}
	char codelet_name[MAX_LINE_SIZE];
	FILE *plt = fopen("data_trace.gp", "w+");
	if(!plt){
		perror("Error while creating data_trace.plt:");
		exit(-1);
	}

	fprintf(plt, "#!/usr/bin/gnuplot -persist\n\n");
	fprintf(plt, "set term postscript eps enhanced color\n");
	fprintf(plt, "set output \"data_trace.eps\"\n");
	fprintf(plt, "set title \"Data trace\"\n");
	fprintf(plt, "set logscale x\n");
	fprintf(plt, "set logscale y\n");
	fprintf(plt, "set xlabel \"tasks size (ms)\"\n");
	fprintf(plt, "set ylabel \"data size (B)\"\n");
	fprintf(plt, "plot ");
	int c_iter;
	char *v_iter;
	int begin = 1;
	while(fgets(codelet_name, MAX_LINE_SIZE, codelet_list) != NULL)
	{
		if(argc == 0)
		{
			if(begin)
				begin = 0;
			else
			fprintf(plt, ", ");
		}
		int size = strlen(codelet_name);
		if(size > 0)
			codelet_name[size-1] = '\0';
		if(argc != 0)
		{
			for(c_iter = 0, v_iter = argv[c_iter];
			    c_iter < argc;
			    c_iter++, v_iter = argv[c_iter])
			{
				if(!strcmp(v_iter, codelet_name))
				{
					if(begin)
						begin = 0;
					else
						fprintf(plt, ", ");
					fprintf(plt, "\"%s\" using 1:2 with dots lw 1 title \"%s\"", codelet_name, codelet_name);
				}
			}
		}
		else
		{
			fprintf(plt, "\"%s\" using 1:2 with dots lw 1 title \"%s\"", codelet_name, codelet_name);
		}
	}
	fprintf(plt, "\n");

	if(fclose(codelet_list))
	{
		perror("close failed :");
		exit(-1);
	}

	if(fclose(plt))
	{
		perror("close failed :");
		exit(-1);
	}
}

int main(int argc, char **argv)
{
	if(argc < 2)
	{
		usage(argv[0]);
	}
	starpu_fxt_write_data_trace(argv[1]);
	write_plt(argc - 2, argv + 2);
	return 0;
}
