#include <starpu.h>

struct starpu_sched_ctx_hypervisor_criteria* sched_ctx_hypervisor_init(void);

void sched_ctx_hypervisor_shutdown(void);

void sched_ctx_hypervisor_handle_ctx(unsigned sched_ctx);

void sched_ctx_hypervisor_ignore_ctx(unsigned sched_ctx);

void sched_ctx_hypervisor_set_resize_interval(unsigned sched_ctx, unsigned min_nprocs, unsigned max_nprocs);

void sched_ctx_hypervisor_set_resize_granularity(unsigned sched_ctx, unsigned granluarity);

void sched_ctx_hypervisor_set_idle_max_value(unsigned sched_ctx, int max_idle_value, int *workers, int nworkers);

void sched_ctx_hypervisor_set_work_min_value(unsigned sched_ctx, int min_working_value, int *workers, int nworkers); 

void sched_ctx_hypervisor_increase_priority(unsigned sched_ctx, int priority_step, int *workers, int nworkers);

void sched_ctx_hypervisor_update_current_idle_time(unsigned sched_ctx, int worker, double idle_time, unsigned current_nprocs);

void sched_ctx_hypervisor_update_current_working_time(unsigned sched_ctx, int worker, double working_time, unsigned current_nprocs);

