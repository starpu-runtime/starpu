#include <stdio.h>
#include <starpu.h>
#include <math.h>

void cpu_mandelbrot(void *descr[], void *cl_arg)
{
        long long int *pixels;
	float *params;

        pixels = (long long int *)STARPU_MATRIX_GET_PTR(descr[0]);
	params = (float *)STARPU_MATRIX_GET_PTR(descr[1]);

        int width = STARPU_MATRIX_GET_NX(descr[0]);
        int height = STARPU_MATRIX_GET_NY(descr[0]);
        
        int ldP = STARPU_MATRIX_GET_LD(descr[0]);

        float centerr = params[0];
        float centeri = params[1];

        float offset = params[2];
        float dim = params[3];
        float zoom = width * 0.25296875;
        float conv_limit = 2.0;
        int max_iter = (width/2) * 0.049715909 * log10(zoom);

        int x,y,n;

        for (y = 0; y < height; y++){
                for (x = 0; x < width; x++){
                        float cr = centerr + (x - (dim/2))/zoom;
                        float ci = centeri + (y+offset - (dim/2))/zoom;
                        float zr = cr;
                        float zi = ci;
                        float m = zr * zr + zi * zi;
                        
                        for (n = 0; n <= max_iter && m < conv_limit * conv_limit; n++) {

                                float tmp = zr*zr - zi*zi + cr;
                                zi = 2*zr*zi + ci;
                                zr = tmp;
                                m = zr*zr + zi*zi;
                        }

		}
		int color;
		if (n==max_iter) fprintf(stderr,".");
		else fprintf(stderr,"%d",n);
		if (n<max_iter)
			color = round(15.*n/max_iter);
		else
			color = 0;
		pixels[x*ldP + y] = color;
	}
}
char* CPU = "cpu_mandelbrot";
char* GPU = "gpu_mandelbrot";
extern char *starpu_find_function(char *name, char *device) {
	if (!strcmp(device,"gpu")) return GPU;
	return CPU;
}
