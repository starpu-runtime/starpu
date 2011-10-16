#include <common/config.h>

#define MAX_IDLE_TIME 5000
#define MIN_WORKING_TIME 500

struct sched_ctx_wrapper {
	/* the sched_ctx it wrappes*/
	unsigned sched_ctx;

	/* underneath this limit we cannot resize */
	unsigned min_nprocs;

	/* above this limit we cannot resize */
	unsigned max_nprocs;
	
	/* priority for a worker to stay in this context */
	/* the smaller the priority the faster it will be moved */
	/* to another context */
	int priority[STARPU_NMAXWORKERS];

	/* above this limit the priority of the worker is reduced */
	double max_idle_time[STARPU_NMAXWORKERS];

	/* underneath this limit the priority of the worker is reduced */
	double min_working_time[STARPU_NMAXWORKERS];

	/* counter for idle time of each worker in a ctx */
	double current_idle_time[STARPU_NMAXWORKERS];

	/* counter for working time of each worker in a ctx */
	double current_working_time[STARPU_NMAXWORKERS];

	/* number of procs of the sched_ctx*/
	unsigned current_nprocs;
};

struct sched_ctx_hypervisor {
	struct sched_ctx_wrapper sched_ctx_wrapper[STARPU_NMAX_SCHED_CTXS];
	int resize_granularity;
	unsigned resize;
	unsigned num_ctxs;
};

struct sched_ctx_hypervisor hypervisor;
