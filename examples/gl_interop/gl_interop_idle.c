/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * This example demonstrates how to use StarPU combined with OpenGL rendering,
 * which needs:
 *
 * - initializing GLUT first,
 * - enabling it at initialization,
 * - running the corresponding CUDA worker in the GLUT thread (here, the main
 *   thread).
 *
 * The difference with gl_interop.c is that this version runs StarPU Tasks in
 * the glut idle handler.
 */

#include <starpu.h>
#include <unistd.h>

#if (defined(STARPU_USE_CUDA) && defined(STARPU_OPENGL_RENDER))
#include <GL/freeglut.h>

void dummy(void *buffers[], void *cl_arg)
{
	float *v = (float *) STARPU_VECTOR_GET_PTR(buffers[0]);

	printf("Codelet running\n");
	cudaMemsetAsync(v, 0, STARPU_VECTOR_GET_NX(buffers[0]) * sizeof(float), starpu_cuda_get_local_stream());
	cudaStreamSynchronize(starpu_cuda_get_local_stream());
	printf("Codelet done\n");
}

struct starpu_codelet cl =
{
	.cuda_funcs = { dummy },
	.nbuffers = 1,
	.modes = { STARPU_W },
};

void foo(void)
{
}

void display(float i)
{
	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(1, 1, 1);
	glBegin(GL_LINES);
	glVertex2f(-i, -i);
	glVertex2f(i, i);
	glEnd();
	glFinish();
	glutPostRedisplay();
}

static int cuda_devices[] = { 0 };
static struct starpu_driver drivers[] =
{
	{ .type = STARPU_CUDA_WORKER }
};

void callback_func(void *foo)
{
	printf("Callback running, rendering\n");
	float i = 1.;
	while (i > 0)
	{
		usleep(100000);
		display(i);
		i -= 0.1;
	}
	printf("rendering done\n");

	/* Tell it was already the last submitted task */
	starpu_drivers_request_termination();

	/* And terminate StarPU */
	starpu_driver_deinit(&drivers[0]);
	starpu_shutdown();
	exit(0);
}

static void idle(void)
{
	starpu_driver_run_once(&drivers[0]);
}
#endif

int main(int argc, char **argv)
{
#if !(defined(STARPU_USE_CUDA) && defined(STARPU_OPENGL_RENDER))
	return 77;
#else
	struct starpu_conf conf;
	int ret;
	struct starpu_task *task;
	starpu_data_handle_t handle;
	int cuda_device = 0;

	cuda_devices[0] = cuda_device;
	drivers[0].id.cuda_id = cuda_device;

	glutInit(&argc, argv);
	glutInitDisplayMode (GLUT_SINGLE | GLUT_RGB);
	glutInitWindowPosition(0, 0);
	glutInitWindowSize(300,200);
	glutCreateWindow("StarPU OpenGL interoperability test");
	glClearColor (0.5, 0.5, 0.5, 0.0);

	/* Enable OpenGL interoperability */
	starpu_conf_init(&conf);
	conf.ncuda = 1;
	conf.ncpus = 0;
	conf.nopencl = 0;
	conf.cuda_opengl_interoperability = cuda_devices;
	conf.n_cuda_opengl_interoperability = sizeof(cuda_devices) / sizeof(*cuda_devices);
	conf.not_launched_drivers = drivers;
	conf.n_not_launched_drivers = sizeof(drivers) / sizeof(*drivers);
	ret = starpu_init(&conf);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_vector_data_register(&handle, -1, 0, 10, sizeof(float));

	/* Submit just one dumb task */
	task = starpu_task_create();
	task->cl = &cl;
	task->handles[0] = handle;
	task->callback_func = callback_func;
	task->callback_arg = NULL;
	ret = starpu_task_submit(task);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	/* And run the driver inside main, which will run the task */
	printf("running the driver\n");
	/* Initialize it */
	starpu_driver_init(&drivers[0]);
	/* Register driver loop content as idle handler */
	glutIdleFunc(idle);
	/* Now run the glut loop */
	glutMainLoop();
	/* And deinitialize driver */
	starpu_driver_deinit(&drivers[0]);
	printf("finished running the driver\n");

	starpu_shutdown();

	return 0;
#endif
}
