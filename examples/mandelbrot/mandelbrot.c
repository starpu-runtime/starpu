/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2011 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <starpu.h>
#ifdef STARPU_USE_OPENCL
#include <starpu_opencl.h>
#endif
#include <sys/time.h>

#ifdef STARPU_HAVE_X11
#include <X11/Xlib.h>
#include <X11/Xutil.h>
int use_x11 = 1;
#endif

int demo = 0;

/* NB: The X11 code is inspired from the http://locklessinc.com/articles/mandelbrot/ article */

static int nblocks = 16;
static int height = 1280;
static int width = 1600;
static int maxIt = 20000;

static double leftX = -0.745;
static double rightX = -0.74375;
static double topY = .15;
static double bottomY = .14875;

/*
 *	X11 window management
 */

#ifdef STARPU_HAVE_X11
/* X11 data */
static Display *dpy;
static Window win;
static XImage *bitmap;
static GC gc;
static KeySym Left=-1, Right, Down, Up, Alt ;

static void exit_x11(void)
{
	XDestroyImage(bitmap);
	XDestroyWindow(dpy, win);
	XCloseDisplay(dpy);
}

static void init_x11(int width, int height, unsigned *buffer)
{
	/* Attempt to open the display */
	dpy = XOpenDisplay(NULL);
	
	/* Failure */
	if (!dpy)
		exit(0);
	
	unsigned long white = WhitePixel(dpy,DefaultScreen(dpy));
	unsigned long black = BlackPixel(dpy,DefaultScreen(dpy));

	win = XCreateSimpleWindow(dpy, DefaultRootWindow(dpy), 0, 0,
					width, height, 0, black, white);
	
	/* We want to be notified when the window appears */
	XSelectInput(dpy, win, StructureNotifyMask);
	
	/* Make it appear */
	XMapWindow(dpy, win);
	
	XTextProperty tp;
	char name[128] = "Mandelbrot - StarPU";
	char *n = name;
	Status st = XStringListToTextProperty(&n, 1, &tp);
	if (st)
		XSetWMName(dpy, win, &tp);

	/* Wait for the MapNotify event */
	XFlush(dpy);
	
	int depth = DefaultDepth(dpy, DefaultScreen(dpy));
	Visual *visual = DefaultVisual(dpy, DefaultScreen(dpy));

	/* Make bitmap */
	bitmap = XCreateImage(dpy, visual, depth,
		ZPixmap, 0, (char *)buffer,
		width, height, 32, 0);
	
	/* Init GC */
	gc = XCreateGC(dpy, win, 0, NULL);
	XSetForeground(dpy, gc, black);
	
	XSelectInput(dpy, win, ExposureMask | KeyPressMask | StructureNotifyMask);
	
	Atom wmDeleteMessage;
	wmDeleteMessage = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
	XSetWMProtocols(dpy, win, &wmDeleteMessage, 1);

        Left = XStringToKeysym ("Left");
        Right = XStringToKeysym ("Right");
        Up = XStringToKeysym ("Up");
        Down = XStringToKeysym ("Down");
        Alt = XStringToKeysym ("Alt");
}

static int handle_events(void)
{
	XEvent event;
	XNextEvent(dpy, &event);

	KeySym key;
	char text[255];

	if (event.type == KeyPress)
	{
		XLookupString(&event.xkey,text,255,&key,0);
		if (key == Left)
		{
			double widthX = rightX - leftX;
			leftX -= 0.25*widthX;
			rightX -= 0.25*widthX;
		}
		else if (key == Right)
		{
			double widthX = rightX - leftX;
			leftX += 0.25*widthX;
			rightX += 0.25*widthX;
		}
		else if (key == Up)
		{
			double heightY = topY - bottomY;
			topY += 0.25*heightY;
			bottomY += 0.25*heightY;
		}
		else if (key == Down)
		{
			double heightY = topY - bottomY;
			topY -= 0.25*heightY;
			bottomY -= 0.25*heightY;
		}
		else {
			double widthX = rightX - leftX;
			double heightY = topY - bottomY;

			if (text[0] == '-')
			{
				/* Zoom out */
				leftX -= 0.125*widthX;
				rightX += 0.125*widthX;
				topY += 0.125*heightY;
				bottomY -= 0.125*heightY;
			}
			else if (text[0] == '+')
			{
				/* Zoom in */
				leftX += 0.125*widthX;
				rightX -= 0.125*widthX;
				topY -= 0.125*heightY;
				bottomY += 0.125*heightY;
			}
		}

		if (text[0]=='q') {
			return -1;
		}
	}

	if (event.type==ButtonPress) {
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
	starpu_unpack_cl_args(cl_arg, &iby, &block_size, &stepX, &stepY);

	cl_mem data = (cl_mem)STARPU_VECTOR_GET_PTR(descr[0]);

	cl_kernel kernel;
	cl_command_queue queue;
	cl_event event;

	int id = starpu_worker_get_id();
	int devid = starpu_worker_get_devid(id);

	starpu_opencl_load_kernel(&kernel, &queue, &opencl_programs, "mandelbrot_kernel", devid);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &data);
	clSetKernelArg(kernel, 1, sizeof(double), &leftX);
	clSetKernelArg(kernel, 2, sizeof(double), &topY);
	clSetKernelArg(kernel, 3, sizeof(double), &stepX);
	clSetKernelArg(kernel, 4, sizeof(double), &stepY);
	clSetKernelArg(kernel, 5, sizeof(int), &maxIt);
	clSetKernelArg(kernel, 6, sizeof(int), &iby);
	clSetKernelArg(kernel, 7, sizeof(int), &block_size);
	clSetKernelArg(kernel, 8, sizeof(int), &width);

	unsigned dim = 16;
	size_t local[2] = {dim, 1};
	size_t global[2] = {width, block_size};
	clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, &event);
	clFinish(queue);
	starpu_opencl_collect_stats(event);
	clReleaseEvent(event);
	starpu_opencl_release_kernel(kernel);
}
#endif

/*
 *	CPU kernel
 */

static void compute_block(void *descr[], void *cl_arg)
{
	int ix, iy;

	int iby, block_size;
	double stepX, stepY;
	starpu_unpack_cl_args(cl_arg, &iby, &block_size, &stepX, &stepY);

	unsigned *data = (unsigned *)STARPU_VECTOR_GET_PTR(descr[0]);

	int local_iy;
	for (local_iy = 0; local_iy < block_size; local_iy++)
	{
		iy = iby*block_size + local_iy;
		for (ix = 0; ix < width; ix++)
		{
			double cx = leftX + ix * stepX;
			double cy = topY - iy * stepY;
			// Z = X+I*Y
			double x = 0;
			double y = 0;
			int it;
			for (it = 0; it < maxIt; it++)
			{
				double x2 = x*x;
				double y2 = y*y;

				// Stop iterations when |Z| > 2
				if (x2 + y2 > 4.0)
					break;

				double twoxy = 2.0*x*y;

				// Z = Z^2 + C
				x = x2 - y2 + cx;
				y = twoxy + cy;
			}
	
			unsigned int v = STARPU_MIN((1024*((float)(it)/(2000))), 256);
			data[ix + local_iy*width] = (v<<16|(255-v)<<8);
		}
	}
}

static starpu_codelet mandelbrot_cl = {
	.where = STARPU_CPU|STARPU_OPENCL,
	.type = STARPU_SEQ,
	.cpu_func = compute_block,
#ifdef STARPU_USE_OPENCL
	.opencl_func = compute_block_opencl,
#endif
	.nbuffers = 1
};

static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-h") == 0) {
			fprintf(stderr, "Usage: %s [-h] [ -width 800] [-height 600] [-nblocks 16] [-no-x11] [-pos leftx:rightx:bottomy:topy]\n", argv[0]);
			exit(-1);
		}

		if (strcmp(argv[i], "-width") == 0) {
			char *argptr;
			width = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-height") == 0) {
			char *argptr;
			height = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-nblocks") == 0) {
			char *argptr;
			nblocks = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-pos") == 0) {
			int ret = sscanf(argv[++i], "%lf:%lf:%lf:%lf", &leftX, &rightX, &bottomY, &topY);
			assert(ret == 4);
		}

		if (strcmp(argv[i], "-demo") == 0) {
			demo = 1;
			leftX = -50.22749575062760;
			rightX = 48.73874621262927;
			topY = -49.35016705749115;
			bottomY = 49.64891691946615;

		}

		if (strcmp(argv[i], "-no-x11") == 0) {
#ifdef STARPU_HAVE_X11
			use_x11 = 0;
#endif
		}
	}
}

int main(int argc, char **argv)
{
	parse_args(argc, argv);

	starpu_init(NULL);

	unsigned *buffer;
	starpu_data_malloc_pinned_if_possible((void **)&buffer, height*width*sizeof(unsigned));

#ifdef STARPU_HAVE_X11
	if (use_x11)
		init_x11(width, height, buffer);
#endif

	int block_size = height/nblocks;
	STARPU_ASSERT((height % nblocks) == 0);

#ifdef STARPU_USE_OPENCL
	starpu_opencl_load_opencl_from_string(mandelbrot_opencl_src, &opencl_programs);
#endif

	starpu_data_handle block_handles[nblocks];
	
	int iby;
	for (iby = 0; iby < nblocks; iby++)
	{
		unsigned *data = &buffer[iby*block_size*width];
		starpu_vector_data_register(&block_handles[iby], 0,
                        (uintptr_t)data, block_size*width, sizeof(unsigned));
	}

	while (1)
	{
		double stepX = (rightX - leftX)/width;
		double stepY = (topY - bottomY)/height;

		struct timeval start, end;

		gettimeofday(&start, NULL);

		for (iby = 0; iby < nblocks; iby++)
		{
			starpu_insert_task(&mandelbrot_cl,
				STARPU_VALUE, &iby, sizeof(iby),
				STARPU_VALUE, &block_size, sizeof(block_size),
				STARPU_VALUE, &stepX, sizeof(stepX),
				STARPU_VALUE, &stepY, sizeof(stepY),
				STARPU_W, block_handles[iby],
				0);
		}

		for (iby = 0; iby < nblocks; iby++)
		{
			starpu_data_acquire(block_handles[iby], STARPU_R);
#ifdef STARPU_HAVE_X11
			if (use_x11)
			{
				XPutImage(dpy, win, gc, bitmap,
					0, iby*block_size,
					0, iby*block_size,
					width, block_size);
			}
#endif
			starpu_data_release(block_handles[iby]);
		}

		gettimeofday(&end, NULL);
		double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

		fprintf(stderr, "Time to generate frame : %f ms\n", timing/1000.0);
		fprintf(stderr, "%14.14f:%14.14f:%14.14f:%14.14f\n", leftX, rightX, bottomY, topY);

#ifdef STARPU_HAVE_X11
		if (demo)
		{
			/* Zoom in */
			double zoom_factor = 0.05;
			double widthX = rightX - leftX;
			double heightY = topY - bottomY;
			leftX += (zoom_factor/2)*widthX;
			rightX -= (zoom_factor/2)*widthX;
			topY -= (zoom_factor/2)*heightY;
			bottomY += (zoom_factor/2)*heightY;
	
		}
		else if (use_x11 && handle_events())
			break;
#endif
	}

#ifdef STARPU_HAVE_X11
	if (use_x11)
		exit_x11();
#endif

	for (iby = 0; iby < nblocks; iby++)
		starpu_data_unregister(block_handles[iby]);

	starpu_data_free_pinned_if_possible(buffer);

	starpu_shutdown();

	return 0;
}
