#include "graph_sched_internal.hpp"

graph_sched_data *graph_sched_graph_policy_data(unsigned sched_ctx_id)
{
    if (sched_ctx_id == 0u)
        sched_ctx_id = starpu_sched_ctx_get_context();
    if (sched_ctx_id >= (unsigned)STARPU_NMAX_SCHED_CTXS)
        sched_ctx_id = 0u;
    struct starpu_sched_policy *pol = starpu_sched_get_sched_policy_in_ctx(sched_ctx_id);
    if (!pol || !pol->policy_name)
        return nullptr;
    if (std::strcmp(pol->policy_name, "sgoc") != 0)
        return nullptr;
    void *p = starpu_sched_ctx_get_policy_data(sched_ctx_id);
    return p ? static_cast<graph_sched_data *>(p) : nullptr;
}
