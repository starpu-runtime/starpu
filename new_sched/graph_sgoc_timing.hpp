/* Phase timing for graph capture / flush (stderr); see STARPU_GRAPH_SCHED_CAPTURE_TIMING. */
#pragma once

#include "graph_sgoc_env.hpp"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>

struct SgocCapturePhaseTimer {
    std::chrono::steady_clock::time_point t;
    const char *const where;
    explicit SgocCapturePhaseTimer(const char *w) : t(std::chrono::steady_clock::now()), where(w) {}
    void lap(const char *label)
    {
        if (!graph_sgoc_bundle::graph_sched_capture_phase_report_enabled())
            return;
        const auto t2 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t2 - t).count();
        t = t2;
        std::cerr << "sgoc_capture_timing: " << where << " +" << std::fixed << std::setprecision(3) << ms << " ms "
                  << label << std::endl;
    }
};
