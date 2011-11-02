#include <starpu.h>
#include <../common/config.h>

/* ioctl properties*/
#define HYPERVISOR_MAX_IDLE 1
#define HYPERVISOR_MIN_WORKING 2
#define HYPERVISOR_PRIORITY 3
#define HYPERVISOR_MIN_PROCS 4
#define HYPERVISOR_MAX_PROCS 5
#define HYPERVISOR_GRANULARITY 6

struct starpu_sched_ctx_hypervisor_criteria* sched_ctx_hypervisor_init(int type);

void sched_ctx_hypervisor_shutdown(void);

void sched_ctx_hypervisor_handle_ctx(unsigned sched_ctx);

void sched_ctx_hypervisor_ignore_ctx(unsigned sched_ctx);

void sched_ctx_hypervisor_resize(unsigned sender_sched_ctx, unsigned receier_sched_ctx, int *workers_to_move, unsigned nworkers_to_movex);

void sched_ctx_hypervisor_set_data(unsigned sched_ctx, void *data);

void* sched_ctx_hypervisor_get_data(unsigned sched_ctx);

void sched_ctx_hypervisor_ioctl(unsigned sched_ctx, ...);

/* hypervisor policies */
#define SIMPLE_POLICY 1

struct hypervisor_policy {
	void (*init)(void);
	void (*deinit)(void);
	void (*add_sched_ctx)(unsigned sched_ctx);
	void(*remove_sched_ctx)(unsigned sched_ctx);
	void (*ioctl)(unsigned sched_ctx, va_list varg_list);
	void (*manage_idle_time)(unsigned req_sched_ctx, int *sched_ctxs, unsigned nsched_ctxs, int worker, double idle_time);
};
