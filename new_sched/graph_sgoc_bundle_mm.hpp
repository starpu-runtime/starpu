#pragma once

#include "graph_sgoc_internal.hpp"

#include <cstdint>
#include <unordered_set>
#include <vector>

/** Cross-TU declarations for graph_sgoc_mm_plan.cpp (MM env, linear planner, offload hints). */
namespace graph_sgoc_bundle {

std::int64_t graph_sgoc_sum_unique_handle_bytes(const std::vector<starpu_data_handle_t> &handles);
int graph_sgoc_mem_offload_auto_env(void);
double graph_sgoc_mem_budget_fraction_env(void);
std::int64_t graph_sgoc_force_mem_budget_bytes_env(void);
std::int64_t graph_sgoc_budget_bytes_env(void);
int graph_sgoc_mm_execute_hints_env(void);
void graph_sgoc_apply_gpu_mm_plan_from_capture(const std::vector<GraphOp> &ops,
                                                const std::vector<GraphHandleAccess> &handle_accesses,
                                                const std::vector<size_t> &topo_order, graph_sgoc_data *policy_data,
                                                const graph_sgoc_captured_handle_groups *captured_for_offload_hints,
                                                int pin_worker, int vb, bool batch_matches_previous_flush,
                                                bool outer_batch0_capture, std::vector<void *> &s_offload_active_out,
                                                const std::unordered_set<void *> *starpu_gpu_resident_truth);

void graph_sgoc_log_mm_plan_advance_debug(size_t topo_slots, const graph_sgoc_gpu_memory_manager &mm);

} /* namespace graph_sgoc_bundle */
