/* In-process regression checks (toy graphs; links a subset of scheduler objects). */
#include "graph_sgoc_common.hpp"
#include "graph_sgoc_bundle_topo.hpp"
#include "graph_sgoc_bundle_checkpoint.hpp"

#include <cstdlib>
#include <iostream>

static int fail(const char *msg)
{
    std::cerr << "graph_sgoc_self_test: " << msg << std::endl;
    return 1;
}

static int test_topological_order()
{
    std::vector<GraphOp> ops(3);
    for (GraphOp &op : ops)
        op.kind = GraphOp::TASK;
    /* 0 -> 1 -> 2 */
    ops[0].successors.push_back(1);
    ops[1].predecessors.push_back(0);
    ops[1].successors.push_back(2);
    ops[2].predecessors.push_back(1);

    std::vector<size_t> order;
    graph_sgoc_bundle::graph_sched_compute_topological_order(ops, order);
    if (order.size() != 3)
        return fail("topo: expected 3 nodes");

    auto pos = [&order](size_t i) {
        for (size_t k = 0; k < order.size(); ++k) {
            if (order[k] == i)
                return k;
        }
        return order.size();
    };
    if (pos(0) >= pos(1) || pos(1) >= pos(2))
        return fail("topo: dependency order violated");
    return 0;
}

static int test_memory_after_ops_empty_handles()
{
    std::vector<GraphOp> ops(1);
    ops[0].kind = GraphOp::TASK;
    std::vector<GraphHandleAccess> ha;
    std::vector<size_t> order = {0};
    size_t peak_i = 0;
    std::int64_t peak_b = -1, init_b = -1;
    size_t init_live = 999;
    graph_sgoc_bundle::graph_sched_compute_memory_after_ops(ops, ha, order, &peak_i, &peak_b, &init_b, &init_live, false);
    if (peak_b < 0 || init_b < 0)
        return fail("memory sim: expected non-negative bytes model");
    if (init_live != 0)
        return fail("memory sim: expected zero initial live handles for empty HA");
    return 0;
}

static int test_checkpoint_templates_empty()
{
    GraphOp op{};
    op.kind = GraphOp::TASK;
    std::vector<SgocWrrCheckpointTemplate> tpl;
    if (graph_sgoc_bundle::graph_sched_op_matches_wrr_checkpoint_templates(op, tpl))
        return fail("checkpoint: empty templates should not match");
    return 0;
}

int main()
{
    if (int e = test_topological_order())
        return e;
    if (int e = test_memory_after_ops_empty_handles())
        return e;
    if (int e = test_checkpoint_templates_empty())
        return e;
    std::cerr << "graph_sgoc_self_test: ok\n";
    return 0;
}
