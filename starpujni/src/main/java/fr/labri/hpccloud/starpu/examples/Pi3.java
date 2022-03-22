package fr.labri.hpccloud.starpu.examples;

import fr.labri.hpccloud.starpu.StarPU;
import fr.labri.hpccloud.starpu.StarPUException;
import fr.labri.hpccloud.starpu.data.DataSet;

import java.io.PrintStream;
import java.util.Random;

public class Pi3 extends Pi1
{
	public static void main(String[] args) throws StarPUException
	{
		int slices = (args.length == 0) ? SLICES : Integer.valueOf(args[0]);
		compute(slices);
	}

	public static void compute(int slices) throws StarPUException
	{
		compute(System.out, slices);
	}

	public static void compute(PrintStream out, int slices) throws StarPUException
	{
		StarPU.init();
		out.println(String.format("PI3[slice size=%s #slices=%s]", SLICE_SIZE, slices));
		int n = SLICE_SIZE * slices;

		DataSet<Integer> arrayHandle = new DataSet<Integer>(n);
		arrayHandle = arrayHandle.partition(slices);
		if(USE_DRAND48)
		{
			arrayHandle = arrayHandle.init((Void) -> {
					double x = StarPU.drand48() * 2 - 1;
					double y = StarPU.drand48() * 2 - 1;
					return (x * x + y * y <= 1) ? 1 : 0;
				});
		}
		else
		{
			final Random rnd = new Random();
			arrayHandle = arrayHandle.init((Void) -> {
					double x = rnd.nextDouble() * 2 - 1;
					double y = rnd.nextDouble() * 2 - 1;
					return (x * x + y * y <= 1) ? 1 : 0;
				});
		}
		//StarPU.taskWaitForAll();
		int sum = arrayHandle.reduce((i1,i2)->i1+i2, 0);
		arrayHandle = null;

		out.println(USE_DRAND48 ? "DRAN48 PRNG" : "JAVA PRNG");
		out.println("v=" + (4.0 * sum/n));
		StarPU.shutdown();
	}
}
