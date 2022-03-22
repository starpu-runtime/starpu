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

import java.io.InputStream;
import java.util.*;
import java.util.function.BiFunction;
import java.util.function.Function;

import static fr.labri.hpccloud.starpu.data.DataHandle.AccessMode.STARPU_R;
import static fr.labri.hpccloud.starpu.data.DataHandle.AccessMode.STARPU_W;

public class DataSet<T>
{
	protected VectorHandle<T> dataHandle;

	protected DataSet<T> newInstance (int size)
	{
		return new DataSet<>(size);
	}

	protected DataSet<T> newInstance (Collection<T> input)
	{
		return new DataSet<>(input);
	}

	public DataSet(int size)
	{
		dataHandle = VectorHandle.<T>register(size);
	}

	public DataSet(Collection<T> input)
	{
		this(input.size());
		int i = 0;
		for (T t : input)
			dataHandle.setValueAt(i++, t);
	}

	public DataSet(Collection<T> input, int nbParts)
	{
		this(input);
		if(nbParts > 1)
			partition(nbParts);
	}

	public int getSize()
	{
		return dataHandle.getSize();
	}

	public DataSet<T> distinct()
	{
		if (dataHandle.getNbChildren() > 0)
		{
			dataHandle.unpartition();
		}

		Set<T> set = new HashSet<T>();
		for(T t : collect())
		{
			set.add(t);
		}
		return newInstance(set);
	}

	public DataSet<T> union(DataSet<T> other) throws StarPUException
	{
		DataSet<T> result = newInstance(getSize()+other.getSize());
		int nbParts = dataHandle.getNbHandles()+other.dataHandle.getNbHandles();
		long tasks[] = new long[nbParts];
		int start = 0;
		int t = 0;
		for(DataHandle hdl : dataHandle.getHandles())
		{
			tasks[t++] = StarPU.submitTask_(new UnionCodelet<>(start), hdl, result.dataHandle);
			start += ((VectorHandle<T>) hdl).getSize();
		}
		for(DataHandle hdl : other.dataHandle.getHandles())
		{
			tasks[t++] = StarPU.submitTask_(new UnionCodelet<>(start), hdl, result.dataHandle);
			start += ((VectorHandle<T>) hdl).getSize();
		}
		StarPU.waitForTasks(tasks);
		result.dataHandle.partition(nbParts);

		return result;
	}

	public DataSet<T> splitByBlocks(int blockSize)
	{
		int sz = getSize();
		if (sz > blockSize)
		{
			partition((sz / blockSize) + (sz % blockSize == 0 ? 0 : 1));
		}
		return this;
	}

	public DataSet<T> partition (int nbParts)
	{
		assert (nbParts <= getSize());
		dataHandle.partition(nbParts);
		return this;
	}

	public static <E> DataSet<E> readFile(InputStream input, Function<String, E> toStringFunc)
	{
		Scanner scanner = new Scanner(input);
		ArrayList<E> content = new ArrayList<>();
		while (scanner.hasNextLine())
		{
			content.add(toStringFunc.apply(scanner.nextLine()));
		}
		scanner.close();
		return new DataSet<>(content);
	}

	protected static <T> void init (DataHandle dataHandle, Function<Void,T> mapFunc) throws StarPUException
	{
		StarPU.mapCodelet(new InitCodelet<>(mapFunc), false, dataHandle, dataHandle);
	}

	public DataSet<T> init (Function<Void,T> mapFunc) throws StarPUException
	{
		init(dataHandle, mapFunc);
		return this;
	}

	static <K> int count(Iterator<K> it)
	{
		int result = 0;
		for(; it.hasNext(); it.next())
			result++;
		return result;
	}

	public <K> DataSet<K> flatMap(Function<T, Iterator<K>> flatMapFunc) throws StarPUException
	{
		int nbElements = map(t-> count(flatMapFunc.apply(t))).reduce((i1,i2)->i1+i2, 0);

		if (dataHandle.getNbChildren() > 0)
			dataHandle.unpartition();

		DataSet<K> result = new DataSet<>(nbElements);
		StarPU.mapCodelet(new FlatMapCodelet<>(flatMapFunc), false, dataHandle, result.dataHandle);

		return result;
	}

	public <K> DataSet<K> map(Function<T,K> mapFunc) throws StarPUException
	{
		DataSet<K> result = new DataSet<>(dataHandle.getSize());
		if (dataHandle.getNbChildren() > 0)
		{
			result.partition(dataHandle.getNbChildren());
		}
		StarPU.mapCodelet(new MapCodelet<>(mapFunc), false, dataHandle, result.dataHandle);

		return result;
	}

	public interface PairFunction<T,K,V> extends Function<T, Tuple2<K,V>>
	{
	}

	public <U, V> DataPairSet<U,V> mapToPair(PairFunction<T,U,V> toPair) throws StarPUException
	{
		DataPairSet<U,V> result = new DataPairSet<U,V>(dataHandle.getSize());
		if (dataHandle.getNbChildren() > 0)
		{
			result.partition(dataHandle.getNbChildren());
		}
		StarPU.mapCodelet(new MapCodelet<>(toPair), false, dataHandle, result.dataHandle);

		return result;
	}

	public <K> K reduce(BiFunction<T,K,K> redFunc, K initialValue) throws StarPUException
	{
		DataSet<K> resultHandle = new DataSet<>(1);
		if (dataHandle.getNbChildren() > 0)
			dataHandle.unpartition();
		StarPU.reduceCodelet(new ReduceCodelet<T,K>(redFunc, initialValue), true, dataHandle,
				     resultHandle.dataHandle);
		K result = resultHandle.dataHandle.getValueAt(0);

		return result;
	}

	public Iterable<T> collect()
	{
		if (dataHandle.getNbChildren() > 0)
			dataHandle.unpartition();
		return new Iterable<T>()
		{
			@Override
			public Iterator<T> iterator()
			{
				return new Iterator<T>()
				{
					int pos = 0;
					@Override
					public boolean hasNext()
					{
						return pos < dataHandle.getSize();
					}

					@Override
					public T next()
					{
						return dataHandle.getValueAt(pos++);
					}
				};
			}
		};
	}

	private static final class MapCodelet<T,K>  extends Codelet
	{
		Function<T, K> mapFunc;

		MapCodelet(Function<T, K> mapFunc)
		{
			this.mapFunc = mapFunc;
		}

		@Override
		public void run(DataHandle[] buffers)
		{
			VectorHandle<T> inputHandle = (VectorHandle<T>) buffers[0];
			int size = inputHandle.getSize();
			VectorHandle<K> outputHandle = (VectorHandle<K>) buffers[1];

			assert(size == outputHandle.getSize());

			for(int i = 0; i < size; i++)
			{
				T t = inputHandle.getValueAt(i);
				K k = mapFunc.apply(t);
				outputHandle.setValueAt(i, k);
			}
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

	private static final class FlatMapCodelet<T,K>  extends Codelet
	{
		Function<T, Iterator<K>> flatMapFunc;

		FlatMapCodelet(Function<T, Iterator<K>> flatMapFunc)
		{
			this.flatMapFunc = flatMapFunc;
		}

		@Override
		public void run(DataHandle[] buffers)
		{
			VectorHandle<T> inputHandle = (VectorHandle<T>) buffers[0];
			int inSize = inputHandle.getSize();
			VectorHandle<K> outputHandle = (VectorHandle<K>) buffers[1];
			int outSize = outputHandle.getSize();
			int k =0;

			for(int i = 0; i < inSize; i++)
			{
				T t = inputHandle.getValueAt(i);
				Iterator<K> it = flatMapFunc.apply(t);
				while (it.hasNext())
				{
					outputHandle.setValueAt(k++, it.next());
				}
			}
			assert (k == outSize);
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

	private static final class InitCodelet<K>  extends Codelet
	{
		Function<Void, K> mapFunc;

		InitCodelet(Function<Void, K> mapFunc)
		{
			this.mapFunc = mapFunc;
		}

		@Override
		public void run(DataHandle[] buffers)
		{
			Void v = null;
			VectorHandle<K> inputHandle = (VectorHandle<K>) buffers[0];
			int size = inputHandle.getSize();
			VectorHandle<K> outputHandle = (VectorHandle<K>) buffers[1];

			assert(size == outputHandle.getSize());

			for(int i = 0; i < size; i++)
			{
				K k = mapFunc.apply(v);
				outputHandle.setValueAt(i, k);
			}
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

	private static final class ReduceCodelet<T,K>  extends Codelet
	{
		BiFunction<T, K, K> reduceFunc;
		K initialValue;

		ReduceCodelet(BiFunction<T,K,K> reduceFunc, K initialValue)
		{
			this.reduceFunc = reduceFunc;
			this.initialValue = initialValue;
		}

		@Override
		public void run(DataHandle[] buffers)
		{
			VectorHandle<T> inputHandle = (VectorHandle<T>) buffers[0];
			int size = inputHandle.getSize();
			VectorHandle<K> accumulatorHandle = (VectorHandle<K>) buffers[1];

			K accumulator = initialValue;
			for(int i = 0; i < size; i++)
			{
				T t = inputHandle.getValueAt(i);
				accumulator = reduceFunc.apply(t, accumulator);
			}
			accumulatorHandle.setValueAt(0, accumulator);
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

	private static final class UnionCodelet<T> extends Codelet
	{
		int outputStartIndex = 0;

		UnionCodelet(int start)
		{
			outputStartIndex = start;
		}

		@Override
		public synchronized void run(DataHandle[] buffers)
		{
			VectorHandle<T> inputHandle = (VectorHandle<T>) buffers[0];
			int size = inputHandle.getSize();
			VectorHandle<T> outputHandle = (VectorHandle<T>) buffers[1];

			for(int i = 0; i < size; i++)
			{
				T t = inputHandle.getValueAt(i);
				outputHandle.setValueAt(outputStartIndex+i, t);
			}
			outputStartIndex += size;
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
}
