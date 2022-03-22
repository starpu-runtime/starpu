package fr.labri.hpccloud.starpu.examples;

import fr.labri.hpccloud.starpu.Codelet;
import fr.labri.hpccloud.starpu.StarPU;
import fr.labri.hpccloud.starpu.data.DataHandle;
import fr.labri.hpccloud.starpu.data.IntegerVectorHandle;

import java.util.Random;

import static fr.labri.hpccloud.starpu.data.DataHandle.AccessMode.*;

public class Pi1
{
	public static final int SLICES = 1000;
	public static final int SLICE_SIZE = 100000;
	public static final boolean USE_DRAND48 = false;

	static final Codelet sampleCircle = new Codelet()
	{
		@Override
		public void run(DataHandle[] buffers)
		{
			IntegerVectorHandle output = (IntegerVectorHandle)buffers[1];
			Random rnd = new Random();
			int n = output.getSize();

			for (int i = 0; i < n; i++)
			{
				double x, y;

				if (USE_DRAND48 == false)
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
				STARPU_SCRATCH, STARPU_RW
			};
		}
	};

	static final Codelet countOccurences = new Codelet()
	{
		@Override
		public void run(DataHandle[] buffers)
		{
			IntegerVectorHandle array = (IntegerVectorHandle)buffers[0];
			IntegerVectorHandle count = (IntegerVectorHandle)buffers[1];
			int n = array.getSize();
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
			IntegerVectorHandle l = (IntegerVectorHandle) buffers[0];
			IntegerVectorHandle r = (IntegerVectorHandle) buffers[1];
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

		@Override
		public String getName()
		{
			return "redux";
		}
	};

	public static void main(String[] args) throws Exception
	{
		int slices = (args.length == 0) ? SLICES : Integer.valueOf(args[0]);

		compute(slices);
	}

	public static void compute (int slices) throws Exception
	{
		StarPU.init();
		System.out.println(String.format("PI1[slice size=%s #slices=%s]", SLICE_SIZE, slices));
		int n = SLICE_SIZE * slices;
		IntegerVectorHandle arrayHandle = IntegerVectorHandle.register(n);
		IntegerVectorHandle countHandle = IntegerVectorHandle.register(slices);
		IntegerVectorHandle sumHandle = IntegerVectorHandle.register(1);

		arrayHandle.partition(slices);
		countHandle.partition(slices);
		sumHandle.setValueAt(0, 0);
		System.out.println("Sample Circle");
		StarPU.mapCodelet(sampleCircle, false, arrayHandle, arrayHandle);
		System.out.println("Count occurrences");
		StarPU.mapCodelet(countOccurences, false, arrayHandle, countHandle);
		System.out.println("Reduce");
		StarPU.reduceCodelet(redux, true, countHandle, sumHandle);

		int sum = sumHandle.getValueAt(0);

		System.out.println(USE_DRAND48 ? "DRAN48 PRNG" : "JAVA PRNG");
		System.out.println("v=" + (4.0 * sum/n));

		arrayHandle.unpartition();
		arrayHandle.unregister();
		countHandle.unregister();
		sumHandle.unregister();

		StarPU.shutdown();
	}
}
