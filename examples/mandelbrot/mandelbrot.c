/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/*
 * This computes the Mandelbrot set: the output image is split in horizontal
 * stripes, which are computed in parallel.  We also make the same computation
 * several times, so that OpenGL interaction allows to browse through the set.
 */

#include <starpu.h>
#include <math.h>
#include <limits.h>
#ifdef STARPU_HAVE_X11
#include <X11/Xlib.h>
#include <X11/Xutil.h>
int use_x11_p = 1;
#endif

#ifdef STARPU_HAVE_HELGRIND_H
#include <valgrind/helgrind.h>
#endif
#ifndef ANNOTATE_HAPPENS_BEFORE
#define ANNOTATE_HAPPENS_BEFORE(obj) ((void)0)
#endif
#ifndef ANNOTATE_HAPPENS_AFTER
#define ANNOTATE_HAPPENS_AFTER(obj) ((void)0)
#endif

int demo_p = 0;
static double demozoom_p = 0.05;

/* NB: The X11 code is inspired from the http://locklessinc.com/articles/mandelbrot/ article */

static int nblocks_p = 20;
static int height_p = 400;
static int width_p = 640;
static int maxIt_p = 20000; /* max number of iteration in the Mandelbrot function */
static int niter_p = -1; /* number of loops in case we don't use X11, -1 means infinite */
static int use_spmd_p = 0;

static double leftX_p = -0.745;
static double rightX_p = -0.74375;
static double topY_p = .15;
static double bottomY_p = .14875;

/*
 *	X11 window management
 */

#ifdef STARPU_HAVE_X11
/* X11 data */
static Display *dpy_p;
static Window win_p;
static XImage *bitmap_p;
static GC gc_p;
static KeySym Left_p=-1, Right_p, Down_p, Up_p, Alt_p;

static void exit_x11(void)
{
	XDestroyImage(bitmap_p);
	XDestroyWindow(dpy_p, win_p);
	XCloseDisplay(dpy_p);
}

static void init_x11(int width, int height, unsigned *buffer)
{
	/* Attempt to open the display */
	dpy_p = XOpenDisplay(NULL);

	/* Failure */
	if (!dpy_p)
		exit(0);

	unsigned long white = WhitePixel(dpy_p, DefaultScreen(dpy_p));
	unsigned long black = BlackPixel(dpy_p, DefaultScreen(dpy_p));

	win_p = XCreateSimpleWindow(dpy_p, DefaultRootWindow(dpy_p), 0, 0,
				    width, height, 0, black, white);

	/* We want to be notified when the window appears */
	XSelectInput(dpy_p, win_p, StructureNotifyMask);

	/* Make it appear */
	XMapWindow(dpy_p, win_p);

	XTextProperty tp;
	char name[128] = "Mandelbrot - StarPU";
	char *n = name;
	Status st = XStringListToTextProperty(&n, 1, &tp);
	if (st)
		XSetWMName(dpy_p, win_p, &tp);

	/* Wait for the MapNotify event */
	XFlush(dpy_p);

	int depth = DefaultDepth(dpy_p, DefaultScreen(dpy_p));
	Visual *visual = DefaultVisual(dpy_p, DefaultScreen(dpy_p));

	/* Make bitmap */
	bitmap_p = XCreateImage(dpy_p, visual, depth,
				ZPixmap, 0, (char *)buffer,
				width, height, 32, 0);

	/* Init GC */
	gc_p = XCreateGC(dpy_p, win_p, 0, NULL);
	XSetForeground(dpy_p, gc_p, black);

	XSelectInput(dpy_p, win_p, ExposureMask | KeyPressMask | StructureNotifyMask);

	Atom wmDeleteMessage;
	wmDeleteMessage = XInternAtom(dpy_p, "WM_DELETE_WINDOW", False);
	XSetWMProtocols(dpy_p, win_p, &wmDeleteMessage, 1);

        Left_p = XStringToKeysym ("Left");
        Right_p = XStringToKeysym ("Right");
        Up_p = XStringToKeysym ("Up");
        Down_p = XStringToKeysym ("Down");
        Alt_p = XStringToKeysym ("Alt");
}

static int handle_events(void)
{
	XEvent event;

	XNextEvent(dpy_p, &event);
	if (event.type == KeyPress)
	{
		KeySym key;
		char text[255];

		XLookupString(&event.xkey,text,255,&key,0);
		if (key == Left_p)
		{
			double widthX = rightX_p - leftX_p;
			leftX_p -= 0.25*widthX;
			rightX_p -= 0.25*widthX;
		}
		else if (key == Right_p)
		{
			double widthX = rightX_p - leftX_p;
			leftX_p += 0.25*widthX;
			rightX_p += 0.25*widthX;
		}
		else if (key == Up_p)
		{
			double heightY = topY_p - bottomY_p;
			topY_p += 0.25*heightY;
			bottomY_p += 0.25*heightY;
		}
		else if (key == Down_p)
		{
			double heightY = topY_p - bottomY_p;
			topY_p -= 0.25*heightY;
			bottomY_p -= 0.25*heightY;
		}
		else
		{
			double widthX = rightX_p - leftX_p;
			double heightY = topY_p - bottomY_p;

			if (text[0] == '-')
			{
				/* Zoom out */
				leftX_p -= 0.125*widthX;
				rightX_p += 0.125*widthX;
				topY_p += 0.125*heightY;
				bottomY_p -= 0.125*heightY;
			}
			else if (text[0] == '+')
			{
				/* Zoom in */
				leftX_p += 0.125*widthX;
				rightX_p -= 0.125*widthX;
				topY_p -= 0.125*heightY;
				bottomY_p += 0.125*heightY;
			}
		}

		if (text[0]=='q')
		{
			return -1;
		}
	}

	if (event.type==ButtonPress)
	{
		/* tell where the mouse Button was Pressed */
		printf("You pressed a button at (%i,%i)\n",
			event.xbutton.x,event.xbutton.y);
	}

	return 0;
}
#endif

/*
 *	OpenCL kernel
 */

#ifdef STARPU_USE_OPENCL
char *mandelbrot_opencl_src = "\
#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n\
#define MIN(a,b) (((a)<(b))? (a) : (b))					\n\
__kernel void mandelbrot_kernel(__global unsigned* a,			\n\
          double leftX, double topY,					\n\
          double stepX, double stepY,					\n\
          int maxIt, int iby, int block_size, int width)		\n\
{									\n\
    size_t id_x = get_global_id(0);	\n\
    size_t id_y = get_global_id(1);	\n\
    if ((id_x < width) && (id_y < block_size))				\n\
    {									\n\
        double xc = leftX + id_x * stepX;				\n\
        double yc = topY - (id_y + iby*block_size) * stepY;		\n\
        int it;								\n\
        double x,y;							\n\
        x = y = (double)0.0;						\n\
        for (it=0;it<maxIt;it++)					\n\
        {								\n\
          double x2 = x*x;						\n\
          double y2 = y*y;						\n\
          if (x2+y2 > 4.0) break;					\n\
          double twoxy = (double)2.0*x*y;				\n\
          x = x2 - y2 + xc;						\n\
          y = twoxy + yc;						\n\
        }								\n\
       unsigned int v = MIN((1024*((float)(it)/(2000))), 256);		\n\
       a[id_x + width * id_y] = (v<<16|(255-v)<<8);			\n\
   }									\n\
}";

static struct starpu_opencl_program opencl_programs;

static void compute_block_opencl(void *descr[], void *cl_arg)
{
	int iby, block_size;
	double stepX, stepY;
	int *pcnt; /* unused for CUDA tasks */
	starpu_codelet_unpack_args(cl_arg, &iby, &block_size, &stepX, &stepY, &pcnt);

	cl_mem data = (cl_mem)STARPU_VECTOR_GET_DEV_HANDLE(descr[0]);

	cl_kernel kernel;
	cl_command_queue queue;
	cl_int err;

	int id = starpu_worker_get_id_check();
	int devid = starpu_worker_get_devid(id);

	err = starpu_opencl_load_kernel(&kernel, &queue, &opencl_programs, "mandelbrot_kernel", devid);
	if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

	clSetKernelArg(kernel, 0, sizeof(data), &data);
	clSetKernelArg(kernel, 1, sizeof(leftX_p), &leftX_p);
	clSetKernelArg(kernel, 2, sizeof(topY_p), &topY_p);
	clSetKernelArg(kernel, 3, sizeof(stepX), &stepX);
	clSetKernelArg(kernel, 4, sizeof(stepY), &stepY);
	clSetKernelArg(kernel, 5, sizeof(maxIt_p), &maxIt_p);
	clSetKernelArg(kernel, 6, sizeof(iby), &iby);
	clSetKernelArg(kernel, 7, sizeof(block_size), &block_size);
	clSetKernelArg(kernel, 8, sizeof(width_p), &width_p);

	unsigned dim = 16;
	size_t local[2] = {dim, 1};
	size_t global[2] = {width_p, block_size};
	err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);
	if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
	starpu_opencl_release_kernel(kernel);
}
#endif

/*
 *	CPU kernel
 */

static void compute_block(void *descr[], void *cl_arg)
{
	int iby, block_size;
	double stepX, stepY;
	int *pcnt; /* unused for sequential tasks */

	starpu_codelet_unpack_args(cl_arg, &iby, &block_size, &stepX, &stepY, &pcnt);

	unsigned *data = (unsigned *)STARPU_VECTOR_GET_PTR(descr[0]);

	int local_iy;
	for (local_iy = 0; local_iy < block_size; local_iy++)
	{
		int ix, iy;

		iy = iby*block_size + local_iy;
		for (ix = 0; ix < width_p; ix++)
		{
			double cx = leftX_p + ix * stepX;
			double cy = topY_p - iy * stepY;
			/* Z = X+I*Y */
			double x = 0;
			double y = 0;
			int it;
			for (it = 0; it < maxIt_p; it++)
			{
				double x2 = x*x;
				double y2 = y*y;

				/* Stop iterations when |Z| > 2 */
				if (x2 + y2 > 4.0)
					break;

				double twoxy = 2.0*x*y;

				/* Z = Z^2 + C */
				x = x2 - y2 + cx;
				y = twoxy + cy;
			}

			unsigned int v = STARPU_MIN((1024*((float)(it)/(2000))), 256);
			data[ix + local_iy*width_p] = (v<<16|(255-v)<<8);
		}
	}
}

static void compute_block_spmd(void *descr[], void *cl_arg)
{

	int iby, block_size;
	double stepX, stepY;
	int *pcnt;
	starpu_codelet_unpack_args(cl_arg, &iby, &block_size, &stepX, &stepY, &pcnt);

	unsigned *data = (unsigned *)STARPU_VECTOR_GET_PTR(descr[0]);

	while (1)
	{
		int ix, iy; /* global coordinates */
		int local_iy; /* current line */

		local_iy = STARPU_ATOMIC_ADD((unsigned int *)pcnt, 1) - 1;
		ANNOTATE_HAPPENS_BEFORE(pcnt);
		if (local_iy >= block_size)
		{
			ANNOTATE_HAPPENS_AFTER(pcnt);
			break;
		}

		iy = iby*block_size + local_iy;

		for (ix = 0; ix < width_p; ix++)
		{
			double cx = leftX_p + ix * stepX;
			double cy = topY_p - iy * stepY;
			/* Z = X+I*Y */
			double x = 0;
			double y = 0;
			int it;
			for (it = 0; it < maxIt_p; it++)
			{
				double x2 = x*x;
				double y2 = y*y;

				/* Stop iterations when |Z| > 2 */
				if (x2 + y2 > 4.0)
					break;

				double twoxy = 2.0*x*y;

				/* Z = Z^2 + C */
				x = x2 - y2 + cx;
				y = twoxy + cy;
			}

			unsigned int v = STARPU_MIN((1024*((float)(it)/(2000))), 256);
			data[ix + local_iy*width_p] = (v<<16|(255-v)<<8);
		}
	}
}



static struct starpu_codelet spmd_mandelbrot_cl =
{
	.type = STARPU_SPMD,
	.max_parallelism = INT_MAX,
	.cpu_funcs = {compute_block_spmd},
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {compute_block_opencl},
	.opencl_flags = {STARPU_OPENCL_ASYNC},
#endif
	.nbuffers = 1
};

static struct starpu_codelet mandelbrot_cl =
{
	.type = STARPU_SEQ,
	.cpu_funcs = {compute_block},
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {compute_block_opencl},
	.opencl_flags = {STARPU_OPENCL_ASYNC},
#endif
	.nbuffers = 1
};

static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-h") == 0)
		{
			fprintf(stderr, "Usage: %s [-h] [ -width 800] [-height 600] [-nblocks 16] [-no-x11] [-pos leftx:rightx:bottomy:topy] [-niter 1000] [-spmd] [-demo] [-demozoom 0.2]\n", argv[0]);
			exit(-1);
		}

		if (strcmp(argv[i], "-width") == 0)
		{
			char *argptr;
			width_p = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-height") == 0)
		{
			char *argptr;
			height_p = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-nblocks") == 0)
		{
			char *argptr;
			nblocks_p = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-niter") == 0)
		{
			char *argptr;
			niter_p = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-pos") == 0)
		{
			int ret = sscanf(argv[++i], "%lf:%lf:%lf:%lf", &leftX_p, &rightX_p,
					 &bottomY_p, &topY_p);
			assert(ret == 4);
		}

		if (strcmp(argv[i], "-demo") == 0)
		{
			demo_p = 1;
			leftX_p = -50.22749575062760;
			rightX_p = 48.73874621262927;
			topY_p = -49.35016705749115;
			bottomY_p = 49.64891691946615;

		}

		if (strcmp(argv[i], "-demozoom") == 0)
		{
			char *argptr;
			demozoom_p = strtof(argv[++i], &argptr);
		}

		if (strcmp(argv[i], "-no-x11") == 0)
		{
#ifdef STARPU_HAVE_X11
			use_x11_p = 0;
#endif
		}

		if (strcmp(argv[i], "-spmd") == 0)
		{
			use_spmd_p = 1;
		}
	}
}

int main(int argc, char **argv)
{
	int ret;

	parse_args(argc, argv);

	/* We don't use CUDA in that example */
	struct starpu_conf conf;
	starpu_conf_init(&conf);
	conf.ncuda = 0;

	if (use_spmd_p)
	{
		conf.sched_policy_name = "peager";
	}

	ret = starpu_init(&conf);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	unsigned *buffer;
	starpu_malloc((void **)&buffer, height_p*width_p*sizeof(unsigned));

#ifdef STARPU_HAVE_X11
	if (use_x11_p)
		init_x11(width_p, height_p, buffer);
#endif

	int block_size = height_p/nblocks_p;
	STARPU_ASSERT((height_p % nblocks_p) == 0);

#ifdef STARPU_USE_OPENCL
	starpu_opencl_load_opencl_from_string(mandelbrot_opencl_src, &opencl_programs, NULL);
#endif

	starpu_data_handle_t block_handles[nblocks_p];

	int iby;
	for (iby = 0; iby < nblocks_p; iby++)
	{
		unsigned *data = &buffer[iby*block_size*width_p];
		starpu_vector_data_register(&block_handles[iby], STARPU_MAIN_RAM,
					    (uintptr_t)data, block_size*width_p, sizeof(unsigned));
	}

	unsigned iter = 0;

	double start, end;

	start = starpu_timing_now();

	while (niter_p-- != 0)
	{
		double stepX = (rightX_p - leftX_p)/width_p;
		double stepY = (topY_p - bottomY_p)/height_p;

		/* In case we have a SPMD task, each worker will grab tasks in
		 * a greedy and select which piece of image to compute by
		 * incrementing a counter shared by all the workers within the
		 * parallel task. */
		int per_block_cnt[nblocks_p];

		starpu_iteration_push(niter_p);

		for (iby = 0; iby < nblocks_p; iby++)
		{
			per_block_cnt[iby] = 0;
			int *pcnt = &per_block_cnt[iby];

			ret = starpu_task_insert(use_spmd_p?&spmd_mandelbrot_cl:&mandelbrot_cl,
						 STARPU_VALUE, &iby, sizeof(iby),
						 STARPU_VALUE, &block_size, sizeof(block_size),
						 STARPU_VALUE, &stepX, sizeof(stepX),
						 STARPU_VALUE, &stepY, sizeof(stepY),
						 STARPU_W, block_handles[iby],
						 STARPU_VALUE, &pcnt, sizeof(int *),
						 STARPU_TAG_ONLY, ((starpu_tag_t)niter_p)*nblocks_p + iby,
						 0);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
		}

		for (iby = 0; iby < nblocks_p; iby++)
		{
#ifdef STARPU_HAVE_X11
			if (use_x11_p)
			{
				starpu_data_acquire(block_handles[iby], STARPU_R);
				XPutImage(dpy_p, win_p, gc_p, bitmap_p,
					  0, iby*block_size,
					  0, iby*block_size,
					  width_p, block_size);
				starpu_data_release(block_handles[iby]);
			}
#endif
		}


		starpu_iteration_pop();
		if (demo_p)
		{
			/* Zoom in */
			double zoom_factor = demozoom_p;
			double widthX = rightX_p - leftX_p;
			double heightY = topY_p - bottomY_p;

			iter++;

			/* If the window is too small, we reset the demo and display some statistics */
			if ((fabs(widthX) < 1e-12) || (fabs(heightY) < 1e-12))
			{
				leftX_p = -50.22749575062760;
				rightX_p = 48.73874621262927;
				topY_p = -49.35016705749115;
				bottomY_p = 49.64891691946615;

				end = starpu_timing_now();
				double timing = end - start;

				fprintf(stderr, "Time to generate %u frames : %f s\n", iter, timing/1000000.0);
				fprintf(stderr, "Average FPS: %f\n", ((double)iter*1e+6)/timing);

				/* Reset counters */
				iter = 0;
				start = starpu_timing_now();
			}
			else
			{
				leftX_p += (zoom_factor/2)*widthX;
				rightX_p -= (zoom_factor/2)*widthX;
				topY_p -= (zoom_factor/2)*heightY;
				bottomY_p += (zoom_factor/2)*heightY;
			}

		}
#ifdef STARPU_HAVE_X11
		else if (use_x11_p && handle_events())
			break;
#endif
	}

#ifdef STARPU_HAVE_X11
	if (use_x11_p)
		exit_x11();
#endif

	for (iby = 0; iby < nblocks_p; iby++)
		starpu_data_unregister(block_handles[iby]);

/*	starpu_data_free_pinned_if_possible(buffer); */

	starpu_shutdown();

	return 0;
}
