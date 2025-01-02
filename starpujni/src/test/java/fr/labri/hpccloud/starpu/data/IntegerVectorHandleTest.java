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

import org.junit.Test;

import java.util.Random;

public class IntegerVectorHandleTest extends VectorHandleTest {
    protected ScalarVectorHandle register(int size) {
        return IntegerVectorHandle.register(size);
    }

    @Override
    @Test
    public void getSizeTest() {
        super.getSizeTest();
    }

    @Test
    public void setGetTest() {
        Random rnd = new Random();
        int checkedSizes[] = {1, 3, 111 };
        for(int sz : checkedSizes) {
            IntegerVectorHandle hdl = (IntegerVectorHandle) register(sz);
            for (int i = 0 ; i < sz; i++) {
                int value = rnd.nextInt();
                hdl.setValueAt(i, value);
                assertEquals(value, hdl.getValueAt(i));
            }
            hdl.unregister();
        }
    }
}
