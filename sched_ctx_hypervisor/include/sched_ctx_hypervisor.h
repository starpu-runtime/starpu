#include <starpu.h>
#include <pthread.h>

/* ioctl properties*/
#define HYPERVISOR_MAX_IDLE -1
#define HYPERVISOR_MIN_WORKING -2
#define HYPERVISOR_PRIORITY -3
#define HYPERVISOR_MIN_WORKERS -4
#define HYPERVISOR_MAX_WORKERS -5
#define HYPERVISOR_GRANULARITY -6
#define HYPERVISOR_FIXED_WORKERS -7
#define HYPERVISOR_MIN_TASKS -8
#define HYPERVISOR_NEW_WORKERS_MAX_IDLE -9
#define HYPERVISOR_TIME_TO_APPLY -10
#define HYPERVISOR_EMPTY_CTX_MAX_IDLE -11

struct sched_ctx_hypervisor_reply{
	int procs[STARPU_NMAXWORKERS];
	int nprocs;
};
pthread_mutex_t act_hypervisor_mutex;

#define MAX_IDLE_TIME 5000000000
#define MIN_WORKING_TIME 500

struct policy_config {
	/* underneath this limit we cannot resize */
	int min_nworkers;

	/* above this limit we cannot resize */
	int max_nworkers;
	
	/*resize granularity */
	int granularity;

	/* priority for a worker to stay in this context */
	/* the smaller the priority the faster it will be moved */
	/* to another context */
	int priority[STARPU_NMAXWORKERS];

	/* above this limit the priority of the worker is reduced */
	double max_idle[STARPU_NMAXWORKERS];

	/* underneath this limit the priority of the worker is reduced */
	double min_working[STARPU_NMAXWORKERS];

	/* workers that will not move */
	int fixed_workers[STARPU_NMAXWORKERS];

	/* max idle for the workers that will be added during the resizing process*/
	double new_workers_max_idle;

	/* above this context we allow removing all workers */
	double empty_ctx_max_idle[STARPU_NMAXWORKERS];
};


struct starpu_sched_ctx_hypervisor_criteria* sched_ctx_hypervisor_init(int type);

void sched_ctx_hypervisor_shutdown(void);

void sched_ctx_hypervisor_handle_ctx(unsigned sched_ctx, double total_flops);

void sched_ctx_hypervisor_ignore_ctx(unsigned sched_ctx);

unsigned sched_ctx_hypervisor_resize(unsigned sched_ctx, int task_tag);

void sched_ctx_hypervisor_move_workers(unsigned sender_sched_ctx, unsigned receier_sched_ctx, int *workers_to_move, unsigned nworkers_to_movex);

void sched_ctx_hypervisor_stop_resize(unsigned sched_ctx);

void sched_ctx_hypervisor_start_resize(unsigned sched_ctx);

void sched_ctx_hypervisor_set_config(unsigned sched_ctx, void *config);

struct policy_config* sched_ctx_hypervisor_get_config(unsigned sched_ctx);

void sched_ctx_hypervisor_ioctl(unsigned sched_ctx, ...);

void sched_ctx_hypervisor_steal_workers(unsigned sched_ctx, int *workers, int nworkers, int task_tag);

int* sched_ctx_hypervisor_get_sched_ctxs();

int sched_ctx_hypervisor_get_nsched_ctxs();

double sched_ctx_hypervisor_get_exp_end(unsigned sched_ctx);

double sched_ctx_hypervisor_get_flops_left_pct(unsigned sched_ctx);

double sched_ctx_hypervisor_get_idle_time(unsigned sched_ctx, int worker);

double sched_ctx_hypervisor_get_bef_res_exp_end(unsigned sched_ctx);

double sched_ctx_hypervisor_get_ctx_velocity(unsigned sched_ctx);

double sched_ctx_hypervisor_get_cpu_velocity(unsigned sched_ctx);

double sched_ctx_hypervisor_get_flops_left(unsigned sched_ctx);

/* hypervisor policies */
#define IDLE_POLICY 1
#define APP_DRIVEN_POLICY 2
#define GFLOPS_RATE_POLICY 3

struct hypervisor_policy {
	void (*manage_idle_time)(unsigned req_sched_ctx, int worker, double idle_time);
	void (*manage_gflops_rate)(unsigned sched_ctx);
	unsigned (*resize)(unsigned sched_ctx, int *sched_ctxs, unsigned nsched_ctxs);
};
