// StarPU --- Runtime system for heterogeneous multicore architectures.
//
// Copyright (C) 2022-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
package fr.labri.hpccloud.starpu;

import fr.labri.hpccloud.starpu.data.DataHandle;
import fr.labri.hpccloud.starpu.data.IntegerVariableHandle;
import fr.labri.hpccloud.starpu.data.IntegerVectorHandle;
import junit.framework.TestCase;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class StarPUTest extends TestCase {

    @BeforeClass
    public static void beforeClass() throws Exception {
        StarPU.init();
    }

    @AfterClass
    public static void afterClass() throws Exception {
        StarPU.shutdown();
    }
    @Test
    public void submitTaskTest() throws StarPUException {
        IntegerVariableHandle hdl1 = IntegerVariableHandle.register();
        IntegerVariableHandle hdl2 = IntegerVariableHandle.register();

        hdl1.setValue(423);
        hdl2.setValue(324);

        System.err.println("hdl1="+hdl1.toString());
        System.err.println("hdl2="+hdl2.toString());

        StarPU.submitTask(new Codelet() {
                              @Override
                              public void run(DataHandle[] buffers) {
                                  for(DataHandle h : buffers) {
                                    System.err.println("Codelet is running on buffer "+h.toString());
                                    IntegerVariableHandle ih = (IntegerVariableHandle) h;
                                    System.err.println("Codelet is running on buffer"+ih.getValue());
                                  }
                              }

                              @Override
                              public DataHandle.AccessMode[] getAccessModes() {
                                  return new DataHandle.AccessMode[] {
                                      DataHandle.AccessMode.STARPU_R,
                                      DataHandle.AccessMode.STARPU_R
                                  };
                              }

                              @Override
                              public String getName() {
                                  return "test codelet";
                              }
                          }
                , true, new DataHandle[] { hdl1, hdl2});
    }


    @Test
    public void taskWaitForAll() throws StarPUException {
        for (int i = 0; i < 100; i++) {
            final int param = i;
            StarPU.submitTask(new Codelet() {
                @Override
                public void run(DataHandle[] buffers) {
                    System.out.println("Task #" + param);
                }

                @Override
                public DataHandle.AccessMode[] getAccessModes() {
                    return new DataHandle.AccessMode[] { };
                }
            }, false);
        }
        StarPU.taskWaitForAll();
    }
}
