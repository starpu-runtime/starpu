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

public final class Tuple3<K, U, V>
{
	private K k;
	private U u;
	private V v;

	public Tuple3(K k, U u, V v)
	{
		this.k = k;
		this.u = u;
		this.v = v;
	}

	public K _1()
	{
		return k;
	}

	public U _2()
	{
		return u;
	}

	public V _3()
	{
		return v;
	}

	@Override
	public boolean equals(Object obj)
	{
		if (obj.getClass().equals(getClass()))
		{
			Tuple3<K,U,V> po = (Tuple3<K,U,V>)obj;
			return po.u.equals(u) && po.v.equals(v) && po.k.equals(k);
		}
		return false;
	}

	public Tuple2<K, Tuple2<U,V>> toKeyPairs()
	{
		return new Tuple2<>(k, new Tuple2<>(u,v));
	}

	@Override
	public int hashCode()
	{
		return Objects.hash(k, u, v);
	}

	@Override
	public String toString()
	{
		return k.toString()+" "+u.toString()+" " + v.toString();
	}
}
