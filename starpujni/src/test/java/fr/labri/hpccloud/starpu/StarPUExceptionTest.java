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
package fr.labri.hpccloud.starpu;

import org.junit.*;

import static org.junit.Assert.*;

public class StarPUExceptionTest {
    private static String message = "message";

    @BeforeClass
    public static void beforeClass() throws Exception {
        StarPU.init();
    }

    @AfterClass
    public static void afterClass() throws Exception {
        StarPU.shutdown();
    }

    @Test(expected = StarPUException.class)
    public void throwStarPUException() throws Exception {
        StarPUException.throwStarPUException(message);
    }

    @Test(expected = StarPUException.class)
    public void throwStarPUExceptionCheckMessage() throws StarPUException {
        try {
            StarPUException.throwStarPUException(message);
        } catch(Exception e) {
            assertEquals(message, e.getMessage());
            throw e;
        }
    }
}