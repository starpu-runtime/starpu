#include <sched_ctx_hypervisor.h>
#include <../common/htable32.h>

struct resize_ack{
	int receiver_sched_ctx;
	int *moved_workers;
	int nmoved_workers;
};

struct sched_ctx_wrapper {
	unsigned sched_ctx;
	void *config;
	double current_idle_time[STARPU_NMAXWORKERS];
	int pushed_tasks[STARPU_NMAXWORKERS];
	int poped_tasks[STARPU_NMAXWORKERS];
	int temp_npushed_tasks;
	int temp_npoped_tasks;
	double total_flops;
	double total_elapsed_flops[STARPU_NMAXWORKERS];
	double elapsed_flops[STARPU_NMAXWORKERS];
	double remaining_flops;
	double start_time;
	double bef_res_exp_end;
	struct resize_ack resize_ack;
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
