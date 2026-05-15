#pragma once

#include "graph_sgoc_internal.hpp"

/** Cross-TU declaration for graph_sgoc_flush.cpp (list rebuild after capture linearize / checkpoint). */
namespace graph_sgoc_bundle {

void graph_sgoc_rebuild_lists_and_refresh_deps(graph_sgoc_data *data);

} /* namespace graph_sgoc_bundle */
