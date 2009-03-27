#ifndef __STARPU_TASK_H__
#define __STARPU_TASK_H__

#include <starpu_config.h>

/* this is a randomly choosen value ... */
#ifndef MAXCUDADEVS
#define MAXCUDADEVS     4
#endif

#ifdef USE_CUDA
#include <cuda.h>
#endif

#include <starpu-data.h>

#define ANY	(~0)
#define CORE	((1ULL)<<1)
#define CUBLAS	((1ULL)<<2)
#define CUDA	((1ULL)<<3)
#define SPU	((1ULL)<<4)
#define GORDON	((1ULL)<<5)

#define MIN_PRIO        (-4)
#define MAX_PRIO        5
#define DEFAULT_PRIO	0

typedef uint64_t tag_t;

/*
 * A codelet describes the various function 
 * that may be called from a worker
 */
typedef struct starpu_codelet_t {
	/* where can it be performed ? */
	uint32_t where;

	/* the different implementations of the codelet */
	void *cuda_func;
	void *cublas_func;
	void *core_func;
	void *spu_func;
	uint8_t gordon_func;

	/* how many buffers do the codelet takes as argument ? */
	unsigned nbuffers;

	struct perfmodel_t *model;
} starpu_codelet;

struct starpu_task {
	struct starpu_codelet_t *cl;

	/* arguments managed by the DSM */
	struct buffer_descr_t buffers[NMAXBUFS];
	data_interface_t interface[NMAXBUFS];

	/* arguments not managed by the DSM are given as a buffer */
	void *cl_arg;
	/* in case the argument buffer has to be uploaded explicitely */
	size_t cl_arg_size;
	
	/* when the task is done, callback_func(callback_arg) is called */
	void (*callback_func)(void *);
	void *callback_arg;

	unsigned use_tag;
	tag_t tag_id;

	/* options for the task execution */
	unsigned synchronous; /* if set, a call to push is blocking */
	int priority; /* MAX_PRIO = most important 
        		: MIN_PRIO = least important */

	/* this is private the StarPU, do not modify */
	void *starpu_private;
};

#ifdef USE_CUDA
/* CUDA specific codelets */
typedef struct cuda_module_s {
	CUmodule module;
	char *module_path;
	unsigned is_loaded[MAXCUDADEVS];
} cuda_module_t;

typedef struct cuda_function_s {
	struct cuda_module_s *module;
	CUfunction function;
	char *symbol;
	unsigned is_loaded[MAXCUDADEVS];
} cuda_function_t;

typedef struct cuda_codelet_s {
	/* which function to execute on the card ? */
	struct cuda_function_s *func;

	/* grid and block shapes */
	unsigned gridx;
	unsigned gridy;
	unsigned blockx;
	unsigned blocky;

	unsigned shmemsize;

	void *stack; /* arguments */
	size_t stack_size;
} cuda_codelet_t;

void init_cuda_module(struct cuda_module_s *module, char *path);
void load_cuda_module(int devid, struct cuda_module_s *module);
void init_cuda_function(struct cuda_function_s *func,
                        struct cuda_module_s *module,
                        char *symbol);
void load_cuda_function(int devid, struct cuda_function_s *function);
#endif // USE_CUDA

/* handle task dependencies: it is possible to associate a task with a unique
 * "tag" and to express dependencies among tasks by the means of those tags */
void tag_remove(tag_t id);
void tag_declare_deps_array(tag_t id, unsigned ndeps, tag_t *array);
void tag_declare_deps(tag_t id, unsigned ndeps, ...);

struct starpu_task *starpu_task_create(void);
int submit_task(struct starpu_task *task);


#endif // __STARPU_TASK_H__
