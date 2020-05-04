#include <stdio.h>
#include <starpu.h>
#include <math.h>

struct params {
        double centerr;
        double centeri;
        long offset;
        long dim;
};


void cpu_mandelbrot(void *descr[], void *cl_arg)
{
        long long *pixels;

        pixels = (long long int *)STARPU_MATRIX_GET_PTR(descr[0]);
        struct params *params = (struct params *) cl_arg;

        long width = STARPU_MATRIX_GET_NY(descr[0]);
        long height = STARPU_MATRIX_GET_NX(descr[0]);
        double zoom = width * 0.25296875;
        double iz = 1. / zoom;
        float diverge = 4.0;
        float max_iterations = (width/2) * 0.049715909 * log10(zoom);
        float imi = 1. / max_iterations;
        double centerr = params->centerr;
        double centeri = params->centeri;
        long offset = params->offset;
        long dim = params->dim;
        double cr = 0;
        double zr = 0;
        double ci = 0;
        double zi = 0;
        long n = 0;
        double tmp = 0;
        int ldP = STARPU_MATRIX_GET_LD(descr[0]);

        long long x,y;

        for (y = 0; y < height; y++){
                for (x = 0; x < width; x++){
                        cr = centerr + (x - (dim/2)) * iz;
			zr = cr;
                        ci = centeri + (y+offset - (dim/2)) * iz;
                        zi = ci;

                        for (n = 0; n <= max_iterations; n++) {
				if (zr*zr + zi*zi>diverge) break;
                                tmp = zr*zr - zi*zi + cr;
                                zi = 2*zr*zi + ci;
                                zr = tmp;
                        }
			if (n<max_iterations)
				pixels[y +x*ldP] = round(15.*n*imi);
			else
				pixels[y +x*ldP] = 0;
		}
	}
}

char* CPU = "cpu_mandelbrot";
char* GPU = "gpu_mandelbrot";
extern char *starpu_find_function(char *name, char *device) {
	if (!strcmp(device,"gpu")) return GPU;
	return CPU;
}
