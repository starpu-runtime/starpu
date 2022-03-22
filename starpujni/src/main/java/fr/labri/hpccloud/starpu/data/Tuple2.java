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

import java.util.Objects;

public final class Tuple2<U, V>
{
	private U u;
	private V v;

	public Tuple2(U u, V v)
	{
		this.u = u;
		this.v = v;
	}
	public U _1()
	{
		return u;
	}

	public V _2()
	{
		return v;
	}

	@Override
	public boolean equals(Object obj)
	{
		if (obj.getClass().equals(getClass()))
		{
			Tuple2<U,V> po = (Tuple2<U,V>)obj;
			return po.u.equals(u) && po.v.equals(v);
		}
		return false;
	}

	@Override
	public int hashCode()
	{
		return Objects.hash(u, v);
	}

	@Override
	public String toString()
	{
		return u.toString()+" " + v.toString();
	}
}
