#include <sched_ctx_hypervisor.h>

struct sched_ctx_wrapper {
	unsigned sched_ctx;
	void *data;
	double current_idle_time[STARPU_NMAXWORKERS];
};

struct sched_ctx_hypervisor {
	struct sched_ctx_wrapper sched_ctx_w[STARPU_NMAX_SCHED_CTXS];
	int sched_ctxs[STARPU_NMAX_SCHED_CTXS];
	unsigned nsched_ctxs;
	unsigned resize;
	struct hypervisor_policy policy;
};

struct sched_ctx_hypervisor hypervisor;
