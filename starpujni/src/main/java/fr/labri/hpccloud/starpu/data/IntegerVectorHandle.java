// StarPU --- Runtime system for heterogeneous multicore architectures.
//
// Copyright (C) 2020-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

public class IntegerVectorHandle extends ScalarVectorHandle
{
	protected IntegerVectorHandle(long handle)
	{
		super(handle);
	}

	public static IntegerVectorHandle register(int size)
	{
		return new IntegerVectorHandle(vectorRegisterInt(size)).setRegistered();
	}

	public int getValueAt(int index)
	{
		return vectorGetIntAt(nativeHandle, index);
	}

	public void setValueAt(int index, int value)
	{
		vectorSetIntAt(nativeHandle, index, value);
	}
}
