// StarPU --- Runtime system for heterogeneous multicore architectures.
//
// Copyright (C) 2022-2024  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
//
// StarPU is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation; either version 2.1 of the License, or (at
// your option) any later version.
//
// StarPU is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
// See the GNU Lesser General Public License in COPYING.LGPL for more details.
//
package fr.labri.hpccloud.yarn.examples.starpu.nativetests;

import fr.labri.hpccloud.starpu.StarPU;
import fr.labri.hpccloud.starpu.StarPUException;
import fr.labri.hpccloud.yarn.MasterSlaveObject;

import java.io.IOException;

public class StarPUMaster extends MasterSlaveObject.Master {
    public StarPUMaster(String prefix) throws IOException {
        super(prefix);
    }

    public void run() {
        out.println("*** Starting native tests on master");
        if (StarPU.runNativeTests()) {
            out.println("SUCCESS");
        } else {
            out.println("FAIL");
        }
        out.println("*** Ending native tests on master");
    }

    public static void main(String[] args) throws StarPUException, IOException {
        StarPU.init();
        StarPUMaster master = new StarPUMaster("master");
        master.run();
        master.terminate();
        StarPU.shutdown();
        System.exit(0);
    }
}
