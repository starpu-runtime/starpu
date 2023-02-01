//! [To be included. You should update doxygen if you see this text.]
package fr.labri.hpccloud.starpu.examples;

import fr.labri.hpccloud.starpu.Codelet;
import fr.labri.hpccloud.starpu.StarPU;
import fr.labri.hpccloud.starpu.data.DataHandle;
import fr.labri.hpccloud.starpu.data.IntegerVariableHandle;
import fr.labri.hpccloud.starpu.data.VectorHandle;

import java.util.Random;

import static fr.labri.hpccloud.starpu.data.DataHandle.AccessMode.*;

public class VectorScal
{
	public static final int NX = 10;
	public static final Float factor = 3.14f;

	static final Codelet scal = new Codelet()
	{
		@Override
		public void run(DataHandle[] buffers)
		{
			VectorHandle<Float> array = (VectorHandle<Float>)buffers[0];
			int n = array.getSize();
			System.out.println(String.format("scaling array %s with %d elements", array, n));
			for (int i = 0; i < n; i++)
			{
				array.setValueAt(i, factor * array.getValueAt(i));
			}
		}

		@Override
		public DataHandle.AccessMode[] getAccessModes()
		{
			return new DataHandle.AccessMode[]
			{
				STARPU_RW
			};
		}
	};

	public static void main(String[] args) throws Exception
	{
		int nx = (args.length == 0) ? NX : Integer.valueOf(args[0]);
		compute(nx);
	}

	public static void compute(int nx) throws Exception
	{
		StarPU.init();
		System.out.println(String.format("VECTOR[#nx=%d]", nx));
		VectorHandle<Float> arrayHandle = VectorHandle.register(nx);
		System.out.println(String.format("scaling array %s", arrayHandle));

		for(int i=0 ; i<nx ; i++)
		{
			arrayHandle.setValueAt(i, i+1.0f);
		}

		StarPU.submitTask(scal, false, arrayHandle);

		arrayHandle.acquire();
		for(int i=0 ; i<nx ; i++)
		{
			System.out.println(String.format("v[%d] = %f", i, arrayHandle.getValueAt(i)));
		}
		arrayHandle.release();

		arrayHandle.unregister();
		StarPU.shutdown();
	}
}
//! [To be included. You should update doxygen if you see this text.]
