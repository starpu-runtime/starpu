#include <stdio.h>
#include <config.h>
#include <starpu.h>

#define PROGNAME "starpu_fxt_data_trace"

static void usage(char *progname)
{
	fprintf(stderr, "Usage : %s <filename>\n", progname);
	exit(77);
}

static void write_plt(){
	FILE *plt = fopen("data_trace.gp", "w+");
	if(!plt){
		fprintf(stderr, "Error while creating data_trace.plt");
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
	fprintf(plt, "plot \"data_total.txt\" using 1:2 with dots lw 1\n");
	if(fclose(plt)){
		perror("close failed :");
		exit(-1);
	}
}

int main(int argc, char **argv)
{
	if(argc != 2)
	{
		usage(argv[0]);
	}
	starpu_fxt_write_data_trace(argv[1]);
	write_plt();
	return 0;
}
