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

import fr.labri.hpccloud.starpu.Codelet;
import fr.labri.hpccloud.starpu.StarPU;
import fr.labri.hpccloud.starpu.StarPUException;

import java.util.*;
import java.util.Map.Entry;
import java.util.function.BiFunction;

import static fr.labri.hpccloud.starpu.data.DataHandle.AccessMode.*;

public class DataPairSet<K, V> extends DataSet<Tuple2<K, V>>
{
	protected DataSet<Tuple2<K, V>> newInstance(int size)
	{
		return new DataPairSet<K, V>(size);
	}

	protected DataSet<Tuple2<K, V>> newInstance(Collection<Tuple2<K, V>> c)
	{
		return new DataPairSet<K, V>(c);
	}

	public DataPairSet(int size)
	{
		super(size);
	}

	public DataPairSet(Collection<Tuple2<K, V>> input)
	{
		super(input);
	}

	public DataPairSet(Map<K, V> input)
	{
		super(input.size());
		int i = 0;
		for (Entry<K, V> e : input.entrySet())
		{
			dataHandle.setValueAt(i++, new Tuple2<>(e.getKey(), e.getValue()));
		}
	}

	public DataPairSet(Collection<Tuple2<K, V>> input, int nbParts)
	{
		super(input, nbParts);
	}

	private static class JoinCodelet<K, V, U> extends Codelet
	{
		<K, V> HashMap<K, Set<V>> handleAsHashMap(VectorHandle<Tuple2<K, V>> hdl)
		{
			int size = hdl.getSize();
			HashMap<K, Set<V>> result = new HashMap<>();
			for (int i = 0; i < size; i++)
			{
				Tuple2<K, V> t = hdl.getValueAt(i);
				Set<V> set = result.get(t._1());
				if (set == null)
				{
					set = new HashSet<>();
					result.put(t._1(), set);
				}
				set.add(t._2());
			}

			return result;
		}

		@Override
		public synchronized void run(DataHandle[] buffers)
		{
			VectorHandle<Tuple2<K, V>> in1 = (VectorHandle<Tuple2<K, V>>) buffers[0];
			VectorHandle<Tuple2<K, U>> in2 = (VectorHandle<Tuple2<K, U>>) buffers[1];
			VectorHandle<ArrayList<Tuple3<K, V, U>>> out = (VectorHandle<ArrayList<Tuple3<K, V, U>>>) buffers[2];
			ArrayList<Tuple3<K, V, U>> res = out.getValueAt(0);
			HashMap<K, Set<V>> map1 = handleAsHashMap(in1);
			HashMap<K, Set<U>> map2 = handleAsHashMap(in2);
			Set<K> keys = map1.keySet();
			keys.retainAll(map2.keySet());

			for (K k : keys)
			{
				for (V v : map1.get(k))
				{
					for (U u : map2.get(k))
					{
						res.add(new Tuple3(k, v, u));
					}
				}
			}
		}

		@Override
		public DataHandle.AccessMode[] getAccessModes()
		{
			return new DataHandle.AccessMode[]
			{
				STARPU_R, STARPU_R, STARPU_W
			};
		}
	}

	public <U> DataPairSet<K, Tuple2<V, U>> join(DataPairSet<K, U> dataPairSet) throws StarPUException
	{
		int nbHandles = dataHandle.getNbHandles();
		VectorHandle<ArrayList<Tuple3<K, V, U>>> tmp = VectorHandle.register(nbHandles);
		if (dataHandle.getNbChildren() > 0)
		{
			tmp.partition(nbHandles);
		}
		init(tmp, (Void) -> new ArrayList<Tuple3<K, V, U>>());
		StarPU.joinCodelet(new JoinCodelet<K, V, U>(), dataHandle.getNbChildren() == 0, dataHandle, dataPairSet.dataHandle, tmp);
		if (dataHandle.getNbChildren() > 0)
		{
			tmp.unpartition();
		}

		int totalSize = 0;
		for (int i = 0; i < nbHandles; i++)
		{
			ArrayList<Tuple3<K, V, U>> l = tmp.getValueAt(i);
			totalSize += l.size();
		}
		DataPairSet<K, Tuple2<V, U>> result = new DataPairSet<>(totalSize);
		int k = 0;
		for (int i = 0; i < nbHandles; i++)
		{
			ArrayList<Tuple3<K, V, U>> l = tmp.getValueAt(i);
			for(int p = 0; p < l.size(); p++)
			{
				Tuple3<K, V, U> t = l.get(p);
				result.dataHandle.setValueAt(k++, t.toKeyPairs());
			}
		}
		tmp.unregister();

		return result;
	}

	private HashMap<K, Set<V>> groupByKeyAsHashMap()
	{
		HashMap<K, Set<V>> resMap = new HashMap<>();
		for (Tuple2<K, V> p : collect())
		{
			Set<V> set = resMap.get(p._1());
			if (set == null)
			{
				set = new HashSet<>();
				resMap.put(p._1(), set);
			}
			set.add(p._2());
		}
		return resMap;
	}

	public DataPairSet<K, Set<V>> groupByKey()
	{
		return new DataPairSet<>(groupByKeyAsHashMap());
	}

	public <U> DataPairSet<K, V> reduceByKey(BiFunction<V, V, V> redFunc) throws StarPUException
	{
		int size = dataHandle.getNbChildren();
		DataSet<HashMap<K, V>> localMaps = new DataSet<>(size == 0 ? 1 : size);
		if (size > 0)
			localMaps.partition(size);

		StarPU.mapCodelet(new CombineByKey(redFunc), true, dataHandle, localMaps.dataHandle);
		if (size > 0)
			localMaps.dataHandle.unpartition();

		HashMap<K, V> resHashmap = localMaps.dataHandle.getValueAt(0);
		for (int i = 1; i < size; i++)
		{
			HashMap<K, V> hm = localMaps.dataHandle.getValueAt(i);
			merge(resHashmap, hm, redFunc);
		}
		DataPairSet<K, V> result = (DataPairSet<K, V>) newInstance(resHashmap.size());
		int i = 0;
		for (Entry<K, V> e : resHashmap.entrySet())
		{
			result.dataHandle.setValueAt(i++, new Tuple2<>(e.getKey(), e.getValue()));
		}

		return result;
	}

	private static final class CombineByKey<K, V> extends Codelet
	{
		private BiFunction<V, V, V> redFunc;

		public CombineByKey(BiFunction<V, V, V> redFunc)
		{
			this.redFunc = redFunc;
		}

		@Override
		public void run(DataHandle[] buffers)
		{
			VectorHandle<Tuple2<K, V>> input = (VectorHandle<Tuple2<K, V>>) buffers[0];
			int inSize = input.getSize();
			VectorHandle<HashMap<K, V>> output = (VectorHandle<HashMap<K, V>>) buffers[1];
			assert (output.getSize() == 1);

			HashMap<K, V> result = new HashMap<>();

			for (int i = 0; i < inSize; i++)
			{
				Tuple2<K, V> p = input.getValueAt(i);
				V v = p._2();
				if (result.containsKey(p._1()))
				{
					V v1 = result.get(p._1());
					v = redFunc.apply(v, v1);
				}
				result.put(p._1(), v);
			}
			output.setValueAt(0, result);
		}

		@Override
		public DataHandle.AccessMode[] getAccessModes()
		{
			return new DataHandle.AccessMode[]
				{
					STARPU_R, STARPU_W
				};
		}
	}

	private static <K, V> void merge(HashMap<K, V> dst, HashMap<K, V> src, BiFunction<V, V, V> redFunc)
	{
		for (Entry<K, V> e : src.entrySet())
		{
			V v = e.getValue();
			if (dst.containsKey(e.getKey()))
			{
				v = redFunc.apply(dst.get(e.getKey()), v);
			}
			dst.put(e.getKey(), v);
		}
	}
}
