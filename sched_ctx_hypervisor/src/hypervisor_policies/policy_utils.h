#include <sched_ctx_hypervisor.h>
#include <pthread.h>

unsigned _find_poor_sched_ctx(unsigned req_sched_ctx, int nworkers_to_move);

int* _get_first_workers(unsigned sched_ctx, unsigned *nworkers, enum starpu_archtype arch);

unsigned _get_potential_nworkers(struct policy_config *config, unsigned sched_ctx, enum starpu_archtype arch);

unsigned _get_nworkers_to_move(unsigned req_sched_ctx);

unsigned _resize(unsigned sender_sched_ctx, unsigned receiver_sched_ctx, unsigned force_resize);

unsigned _resize_to_unknown_receiver(unsigned sender_sched_ctx);
