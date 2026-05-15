#pragma once

#include "graph_sgoc_internal.hpp"

#include <cstdint>
#include <unordered_set>
#include <vector>

struct starpu_task;

/** Cross-TU declarations for graph_sgoc_checkpoint_wrr.cpp (used by graph_sgoc_checkpoint.cpp orchestration). */
namespace graph_sgoc_bundle {

const GraphOpHandleAccessRef *graph_op_find_single_pure_write_access(const GraphOp &op);
bool graph_sgoc_op_matches_wrr_checkpoint_templates(const GraphOp &op,
                                                    const std::vector<SgocWrrCheckpointTemplate> &templates);
void graph_sgoc_collect_wrr_checkpoint_templates(const std::vector<GraphOp> &ops,
                                                 const graph_sgoc_captured_handle_groups &parsed,
                                                 const std::unordered_set<void *> &checkpointable_activation_keys,
                                                 std::vector<SgocWrrCheckpointTemplate> &out);
void graph_sgoc_apply_wrr_checkpoint_templates(std::vector<GraphOp> &ops,
                                               const std::vector<SgocWrrCheckpointTemplate> &templates);
bool graph_sgoc_checkpoint_wrr_chain_resolve(const GraphOp &producer_op, const std::vector<GraphOp> &ops,
                                             const std::vector<GraphHandleAccess> &handle_accesses,
                                             const GraphOpHandleAccessRef **write_ref_out,
                                             std::vector<size_t> &read_accesses_out, const char **failure_reason_out,
                                             unsigned *consecutive_pure_read_tasks_out,
                                             const graph_sgoc_wrr_activation_sub_policy *activation_sub_policy);
struct starpu_task *graph_sgoc_clone_task_for_checkpoint(const struct starpu_task *task);
void graph_sgoc_destroy_checkpoint_task(struct starpu_task *task);
bool graph_sgoc_insert_checkpoint_for_wrr_task(std::vector<GraphOp> &ops,
                                               std::vector<GraphHandleAccess> &handle_accesses, size_t op_idx,
                                               struct starpu_task *checkpoint_task, int pin_worker,
                                               const graph_sgoc_wrr_activation_sub_policy *activation_sub_policy,
                                               size_t *invalidate_insert_pos_out = nullptr);

} /* namespace graph_sgoc_bundle */
