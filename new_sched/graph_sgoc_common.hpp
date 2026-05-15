/* Shared #includes for graph_sgoc_*.cpp (scheduler algorithms split across TUs). */
#pragma once

#include "graph_sched_internal.hpp"
#include "graph_sgoc_env.hpp"
#include "graph_sgoc_timing.hpp"

#include <starpu_graph_capture.h>
#include <starpu_data.h>
#include <starpu_sched_ctx.h>
#include <starpu_scheduler.h>
#include <starpu_stdlib.h>
#include <starpu_task.h>
#include <starpu_task_util.h>
#include <starpu_worker.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
