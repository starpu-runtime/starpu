package fr.labri.hpccloud.starpu.examples;

import fr.labri.hpccloud.starpu.Codelet;
import fr.labri.hpccloud.starpu.StarPU;
import fr.labri.hpccloud.starpu.data.DataHandle;
import fr.labri.hpccloud.starpu.data.IntegerVariableHandle;
import fr.labri.hpccloud.starpu.data.VectorHandle;

import java.util.Random;

import static fr.labri.hpccloud.starpu.data.DataHandle.AccessMode.*;

public class Pi2 extends Pi1
{
	static final Codelet sampleCircle = new Codelet()
	{
		@Override
		public void run(DataHandle[] buffers)
		{
			VectorHandle<Integer> output = (VectorHandle<Integer>)buffers[1];
			Random rnd = new Random();
			int n = output.getSize();

			for (int i = 0; i < n; i++)
			{
				double x, y;

				if (USE_DRAND48== false)
				{
					x = rnd.nextDouble() * 2 - 1;
					y = rnd.nextDouble() * 2 - 1;
				}
				else
				{
					x = StarPU.drand48() * 2 - 1;
					y = StarPU.drand48() * 2 - 1;
				}
				output.setValueAt(i, (x * x + y * y <= 1.0) ? 1 : 0);
			}
		}

		@Override
		public DataHandle.AccessMode[] getAccessModes()
		{
			return new DataHandle.AccessMode[]
			{
				STARPU_SCRATCH, STARPU_W
			};
		}
	};

	static final Codelet countOccurences = new Codelet()
	{
		@Override
		public void run(DataHandle[] buffers)
		{
			VectorHandle<Integer> array = (VectorHandle<Integer>)buffers[0];
			VectorHandle<Integer> count = (VectorHandle<Integer>)buffers[1];
			int n = array.getSize();
			assert(count.getSize() == 1);
			int c = 0;
			for (int i = 0; i < n; i++)
			{
				c += array.getValueAt(i);
			}
			count.setValueAt(0, c);
		}

		@Override
		public DataHandle.AccessMode[] getAccessModes()
		{
			return new DataHandle.AccessMode[]
			{
				STARPU_R, STARPU_RW
			};
		}
	};

	static final Codelet redux = new Codelet()
	{
		@Override
		public void run(DataHandle[] buffers)
		{
			VectorHandle<Integer> l = (VectorHandle<Integer>) buffers[0];
			VectorHandle<Integer> r = (VectorHandle<Integer>) buffers[1];
			int llen = l.getSize();
			int sum = 0;

			for(int i = 0; i < llen; i++)
				sum += l.getValueAt(i);
			r.setValueAt(0, sum);
		}

		@Override
		public DataHandle.AccessMode[] getAccessModes()
		{
			return new DataHandle.AccessMode[]
			{
				DataHandle.AccessMode.STARPU_R,
				DataHandle.AccessMode.STARPU_RW
			};
		}
	};

	public static void main(String[] args) throws Exception
	{
		int slices = (args.length == 0) ? SLICES : Integer.valueOf(args[0]);

		compute(slices);
	}

	public static void compute(int slices) throws Exception
	{
		StarPU.init();
		System.out.println(String.format("PI2[slice size=%s #slices=%s]", SLICE_SIZE, slices));
		int n = SLICE_SIZE * slices;
		VectorHandle<Integer> arrayHandle = VectorHandle.register(n);

		arrayHandle.partition(slices);
		System.out.println("Sample Circle");
		StarPU.mapCodelet(sampleCircle, false, arrayHandle, arrayHandle);

		System.out.println("Count occurrences");
		VectorHandle<Integer> countHandle = VectorHandle.register(slices);
		countHandle.partition(slices);
		StarPU.mapCodelet(countOccurences, false, arrayHandle, countHandle);
		arrayHandle.unpartition();
		arrayHandle.unregister();

		System.out.println("Reduce");
		VectorHandle<Integer> sumHandle = VectorHandle.register(1);
		sumHandle.setValueAt(0, 0);
		StarPU.reduceCodelet(redux, true, countHandle, sumHandle);

		countHandle.unregister();

		int sum = sumHandle.getValueAt(0);

		System.out.println(USE_DRAND48 ? "DRAN48 PRNG" : "JAVA PRNG");
		System.out.println("v=" + (4.0 * sum/n));
		sumHandle.unregister();
		StarPU.shutdown();
	}
}
