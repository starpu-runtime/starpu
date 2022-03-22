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

public class IntegerVariableHandle<T> extends DataHandle
{
	protected IntegerVariableHandle(long handle)
	{
		super(handle);
	}

	public static IntegerVariableHandle register()
	{
		return new IntegerVariableHandle(variableRegisterInt()).setRegistered();
	}

	public int getValue()
	{
		return variableGetIntValue(nativeHandle);
	}

	public void setValue(int v)
	{
		variableSetIntValue(nativeHandle, v);
	}
}
