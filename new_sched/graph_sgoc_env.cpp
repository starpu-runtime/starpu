#include "graph_sgoc_env.hpp"

#include <cstdlib>
#include <cstring>

namespace graph_sgoc_bundle {

int graph_sched_verbose_env(void)
{
    const char *e = std::getenv("STARPU_GRAPH_SCHED_VERBOSE");
    return (e && e[0]) ? std::atoi(e) : 0;
}

bool graph_sched_capture_phase_report_enabled(void)
{
    if (graph_sched_verbose_env() >= 2)
        return true;
    const char *e = std::getenv("STARPU_GRAPH_SCHED_CAPTURE_TIMING");
    return e && e[0] && std::atoi(e) != 0;
}

} /* namespace graph_sgoc_bundle */
