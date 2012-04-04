#include <sched_ctx_hypervisor.h>
#include <common/htable32.h>

struct sched_ctx_hypervisor {
	struct sched_ctx_wrapper sched_ctx_w[STARPU_NMAX_SCHED_CTXS];
	int sched_ctxs[STARPU_NMAX_SCHED_CTXS];
	unsigned nsched_ctxs;
	unsigned resize[STARPU_NMAX_SCHED_CTXS];
	int min_tasks;
	struct hypervisor_policy policy;
	struct starpu_htbl32_node *configurations[STARPU_NMAX_SCHED_CTXS];
	struct starpu_htbl32_node *resize_requests[STARPU_NMAX_SCHED_CTXS];
};

struct sched_ctx_hypervisor_adjustment {
	int workerids[STARPU_NMAXWORKERS];
	int nworkers;
};

struct sched_ctx_hypervisor hypervisor;


void _add_config(unsigned sched_ctx);

void _remove_config(unsigned sched_ctx);
