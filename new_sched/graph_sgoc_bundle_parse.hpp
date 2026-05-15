#pragma once

#include "graph_sgoc_internal.hpp"

#include <cstdint>
#include <vector>

/** Cross-TU declarations for graph_sgoc_parse.cpp (captured-handle groups, batch/subiter classification). */
namespace graph_sgoc_bundle {

void graph_sgoc_parse_captured_data_handles(const std::vector<GraphOp> &ops, graph_sgoc_captured_handle_groups &out,
                                             int verbose);
bool graph_sgoc_infer_batch_capture_context(const std::vector<GraphOp> &ops, bool *has_batch_tags_out,
                                            std::uint32_t *batch_val_out);

bool graph_sgoc_graph_subiter_is_training_stage(std::uint32_t sub);
bool graph_sgoc_graph_subiter_is_forward_stage(std::uint32_t sub);
bool graph_sgoc_graph_subiter_is_backward_stage(std::uint32_t sub);

} /* namespace graph_sgoc_bundle */
