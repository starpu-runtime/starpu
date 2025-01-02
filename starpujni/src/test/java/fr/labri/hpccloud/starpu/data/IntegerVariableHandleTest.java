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
package fr.labri.hpccloud.starpu.data;

import fr.labri.hpccloud.starpu.StarPUTest;
import org.junit.Test;

import static org.junit.Assert.*;

public class IntegerVariableHandleTest extends StarPUTest {
    @Test
    public void registerUnregisterTest() {
        IntegerVariableHandle var = IntegerVariableHandle.register();
        var.unregister();
    }

    @Test
    public void setGetTest() {
        int checkedValues[] = { 12345, 987654 };
        IntegerVariableHandle var = IntegerVariableHandle.register();
        for (int val: checkedValues) {
            var.setValue(val);
            assertEquals(val, var.getValue());
        }
        var.unregister();
    }
}
