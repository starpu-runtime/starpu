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

struct starpu_sched_ctx_hypervisor_criteria* sched_ctx_hypervisor_init(int type);

void sched_ctx_hypervisor_shutdown(void);

void sched_ctx_hypervisor_handle_ctx(unsigned sched_ctx);

void sched_ctx_hypervisor_ignore_ctx(unsigned sched_ctx);

unsigned sched_ctx_hypervisor_resize(unsigned sched_ctx, int task_tag);

void sched_ctx_hypervisor_move_workers(unsigned sender_sched_ctx, unsigned receier_sched_ctx, int *workers_to_move, unsigned nworkers_to_movex);

void sched_ctx_hypervisor_stop_resize(unsigned sched_ctx);

void sched_ctx_hypervisor_start_resize(unsigned sched_ctx);

void sched_ctx_hypervisor_set_config(unsigned sched_ctx, void *config);

void* sched_ctx_hypervisor_get_config(unsigned sched_ctx);

void sched_ctx_hypervisor_ioctl(unsigned sched_ctx, ...);

void sched_ctx_hypervisor_steal_workers(unsigned sched_ctx, int *workers, int nworkers, int task_tag);

int* sched_ctx_hypervisor_get_sched_ctxs();

int sched_ctx_hypervisor_get_nsched_ctxs();

double sched_ctx_hypervisor_get_debit(unsigned sched_ctx);

/* hypervisor policies */
#define SIMPLE_POLICY 1

struct hypervisor_policy {
	void (*init)(void);
	void (*deinit)(void);
	void (*add_sched_ctx)(unsigned sched_ctx);
	void(*remove_sched_ctx)(unsigned sched_ctx);
	void* (*ioctl)(unsigned sched_ctx, va_list varg_list, unsigned later);
	void (*manage_idle_time)(unsigned req_sched_ctx, int worker, double idle_time);
	void (*manage_task_flux)(unsigned sched_ctx);
	unsigned (*resize)(unsigned sched_ctx, int *sched_ctxs, unsigned nsched_ctxs);
	void (*update_config)(void* old_config, void* new_config);
};
