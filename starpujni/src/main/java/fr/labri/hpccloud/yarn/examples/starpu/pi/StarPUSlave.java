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
package fr.labri.hpccloud.yarn.examples.starpu.pi;

import fr.labri.hpccloud.starpu.StarPU;
import fr.labri.hpccloud.starpu.StarPUException;
import fr.labri.hpccloud.starpu.examples.Pi3;
import fr.labri.hpccloud.yarn.MasterSlaveObject;
import fr.labri.hpccloud.yarn.Utils;

import java.io.IOException;

public class StarPUSlave extends MasterSlaveObject.Slave {
    public StarPUSlave(String prefix) throws IOException {
        super(prefix);
    }

    public void run() throws StarPUException, IOException {
        out.println("I'm the StarPU slave "+slaveID);
        out.println("I'm in PWD="+System.getenv("PWD"));
        int nbSlices = Integer.parseInt(Utils.getMandatoryEnv("NB_SLICES"));
        Pi3.compute(out, nbSlices);
    }

    public static void main(String[] args) throws StarPUException, IOException {
        StarPUSlave slave = new StarPUSlave("pi-slave-");
        slave.run();
        slave.terminate();
    }
}
