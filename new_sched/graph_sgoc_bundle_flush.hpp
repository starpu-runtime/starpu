#pragma once

#include "graph_sched_internal.hpp"

/** Cross-TU declaration for graph_sgoc_flush.cpp (list rebuild after capture linearize / checkpoint). */
namespace graph_sgoc_bundle {

void graph_sgoc_rebuild_lists_and_refresh_deps(graph_sched_data *data);

} /* namespace graph_sgoc_bundle */
