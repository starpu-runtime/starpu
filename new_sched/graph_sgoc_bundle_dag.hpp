#pragma once

#include "graph_sgoc_internal.hpp"

#include <cstdint>
#include <unordered_map>
#include <vector>

struct starpu_task;

/** Cross-TU declarations for graph_sgoc_dag.cpp (edges, access modes, capture registration, predicted time). */
namespace graph_sgoc_bundle {

void graph_sgoc_bump_indices_after_insert(graph_sgoc_data *data, size_t insert_pos);
void graph_sgoc_bump_handle_access_op_indices_after_insert(std::vector<GraphHandleAccess> &handle_accesses,
                                                             size_t insert_pos);
void graph_sgoc_bump_op_graph_indices_after_insert(std::vector<GraphOp> &ops, size_t insert_pos);
void graph_sgoc_refresh_op_dependencies(graph_sgoc_data *data);
size_t graph_sgoc_append_handle_access(graph_sgoc_data *data, size_t op_idx, starpu_data_handle_t handle,
                                       unsigned mode, struct starpu_task *task);
void graph_sgoc_register_task_accesses_op(graph_sgoc_data *data, size_t op_idx, struct starpu_task *task, GraphOp &op);
void graph_sgoc_register_task_accesses(graph_sgoc_data *data, size_t op_idx, struct starpu_task *task);
void graph_sgoc_register_invalidate_access_op(graph_sgoc_data *data, GraphOp &op, size_t op_idx,
                                               starpu_data_handle_t handle);
void graph_sgoc_register_invalidate_access(graph_sgoc_data *data, size_t op_idx, starpu_data_handle_t handle);
void graph_sgoc_capture_add_edges_for_op(graph_sgoc_data *data, GraphOp &op);
void graph_sgoc_insert_missing_pre_write_invalidates(graph_sgoc_data *data, struct starpu_task *task);
void graph_sgoc_graph_op_set_stage_from_sched_ctx(GraphOp &op, unsigned task_sched_ctx_id, struct starpu_task *);
unsigned graph_sgoc_iteration_source_sched_ctx(unsigned task_sched_ctx_id);

bool graph_access_mode_is_invalidate(unsigned mode);
bool graph_access_mode_is_pure_read(unsigned mode);
bool graph_access_mode_is_pure_write(unsigned mode);
bool graph_access_mode_is_pure_scratch(unsigned mode);
bool graph_access_mode_is_read_write(unsigned mode);
bool graph_access_mode_is_writer(unsigned mode);
bool graph_access_is_handle_producer_for_deps(const GraphHandleAccess &a);
void graph_op_add_edge(std::vector<GraphOp> &ops, size_t consumer_op_idx, size_t producer_op_idx);
void graph_op_add_edge_stable_sgoc(graph_sgoc_data *data, GraphOp *consumer, size_t producer_stable);
void graph_op_add_pure_read_dependencies_sgoc(graph_sgoc_data *data, GraphOp *consumer, size_t access_idx);
void graph_op_add_writer_or_invalidate_dependencies_sgoc(graph_sgoc_data *data, GraphOp *consumer, size_t access_idx);
void graph_op_add_pure_read_dependencies(graph_sgoc_data *data, size_t consumer_op_idx, size_t access_idx);
void graph_op_add_writer_or_invalidate_dependencies(graph_sgoc_data *data, size_t consumer_op_idx, size_t access_idx);
void graph_sgoc_validate_invalidate_then_pure_write_windows(graph_sgoc_data *data);
bool graph_sgoc_auto_invalidate_enabled(void);
unsigned graph_sgoc_checkpoint_max_env(void);
bool graph_sgoc_linear_replay_greedy_enabled(void);
bool graph_sgoc_task_runnable_on_pinned_worker(const struct starpu_task *task, unsigned workerid);
void graph_sgoc_apply_replay_worker_pin(struct starpu_task *task, int pin_worker, int sched_verbose,
                                         std::unordered_map<const struct starpu_codelet *, bool> *cl_runnable_cache);
double graph_sgoc_predicted_exec_time_us_for_pinned_worker(struct starpu_task *task, int pin_worker,
                                                            unsigned task_sched_ctx_id);
double graph_sgoc_effective_predicted_us(double starpu_expected_length_us);

std::int64_t graph_sgoc_op_intrinsic_memory_delta(const GraphOp &op);

} /* namespace graph_sgoc_bundle */
