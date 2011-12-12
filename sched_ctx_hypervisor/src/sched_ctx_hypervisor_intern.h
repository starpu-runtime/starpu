#include <sched_ctx_hypervisor.h>

struct sched_ctx_wrapper {
	unsigned sched_ctx;
	void *config;
	double current_idle_time[STARPU_NMAXWORKERS];
	int tasks[STARPU_NMAXWORKERS];
	int poped_tasks[STARPU_NMAXWORKERS];
};

struct sched_ctx_hypervisor {
	struct sched_ctx_wrapper sched_ctx_w[STARPU_NMAX_SCHED_CTXS];
	int sched_ctxs[STARPU_NMAX_SCHED_CTXS];
	unsigned nsched_ctxs;
	unsigned resize[STARPU_NMAX_SCHED_CTXS];
	int min_tasks;
	struct hypervisor_policy policy;
	struct starpu_htbl32_node_s *configurations[STARPU_NMAX_SCHED_CTXS];
	struct starpu_htbl32_node_s *steal_requests[STARPU_NMAX_SCHED_CTXS];
	struct starpu_htbl32_node_s *resize_requests[STARPU_NMAX_SCHED_CTXS];
};

struct sched_ctx_hypervisor_adjustment {
	int workerids[STARPU_NMAXWORKERS];
	int nworkers;
};

struct sched_ctx_hypervisor hypervisor;
