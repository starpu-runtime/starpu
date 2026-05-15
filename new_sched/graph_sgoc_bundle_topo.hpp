#pragma once

#include "graph_sgoc_internal.hpp"

#include <cstdint>
#include <unordered_set>
#include <vector>

/** Cross-TU declarations for graph_sgoc_topo.cpp (topological orders, memory simulation, VRAM-ready topo). */
namespace graph_sgoc_bundle {

void graph_sgoc_compute_topological_order(const std::vector<GraphOp> &ops, std::vector<size_t> &order_out);
void graph_sgoc_collect_unique_handles(const std::vector<GraphHandleAccess> &ha,
                                        std::vector<starpu_data_handle_t> &handles_out);
bool graph_sgoc_handle_live_before_graph(const std::vector<GraphHandleAccess> &ha, starpu_data_handle_t h);
std::int64_t graph_sgoc_op_memory_delta_for_resident(const GraphOp &op, const std::unordered_set<void *> &resident);
void graph_sgoc_op_apply_memory_effect_to_resident(const GraphOp &op, std::unordered_set<void *> &resident);
void graph_sgoc_compute_memory_after_ops(const std::vector<GraphOp> &ops, const std::vector<GraphHandleAccess> &ha,
                                         const std::vector<size_t> &topo_order, size_t *peak_topo_index_out,
                                         std::int64_t *peak_bytes_out, std::int64_t *initial_bytes_out,
                                         size_t *initial_live_handle_count_out, bool print_memory_trace);
void graph_sgoc_compute_greedy_memory_topological_order(const std::vector<GraphOp> &ops, std::vector<size_t> &order_out,
                                                         double *greedy_attempt_sec_out, double *lex_fallback_sec_out,
                                                         double *greedy_prep_sec_out, double *greedy_loop_sec_out,
                                                         const std::vector<unsigned> *tie_break);

bool graph_sgoc_ready_vram_topo_enabled(void);
void graph_sgoc_compute_ready_set_greedy_vram_topological_order(const std::vector<GraphOp> &ops,
                                                                 const std::vector<GraphHandleAccess> &handle_accesses,
                                                                 const std::unordered_set<void *> *starpu_gpu_resident_truth,
                                                                 std::vector<size_t> &order_out, int verbose);

} /* namespace graph_sgoc_bundle */
