package fr.labri.hpccloud.starpu.examples;

import fr.labri.hpccloud.starpu.StarPU;
import fr.labri.hpccloud.starpu.data.DataPairSet;
import fr.labri.hpccloud.starpu.data.Tuple2;

import java.util.*;

public class TransitiveClosure
{
	public static final int TC_SEED= 42;
	public static final int TC_NUM_EDGES = 1000;
	public static final int TC_NUM_VERTICES = 500;
	public static final int TC_NUM_SLICES = 5;

	static List<Tuple2<Integer, Integer>> generateGraph(int seed, int numVertices, int numEdges)
	{
		Random rand = new Random(42);
		Set<Tuple2<Integer, Integer>> edges = new HashSet<>(numEdges);
		while (edges.size() < numEdges)
		{
			int from = rand.nextInt(numVertices);
			int to = rand.nextInt(numVertices);
			Tuple2<Integer, Integer> e = new Tuple2<>(from, to);
			if (from != to)
			{
				edges.add(e);
			}
		}
		return new ArrayList<>(edges);
	}

	static class ProjectFn implements DataPairSet.PairFunction<Tuple2<Integer, Tuple2<Integer, Integer>>, Integer, Integer>
	{
		static final ProjectFn INSTANCE = new ProjectFn();

		@Override
		public Tuple2<Integer, Integer> apply(Tuple2<Integer, Tuple2<Integer, Integer>> triple)
		{
			return new Tuple2<>(triple._2()._2(), triple._2()._1());
		}
	}

	public static void main(String[] args) throws Exception
	{
		compute(TC_SEED, TC_NUM_VERTICES, TC_NUM_EDGES, TC_NUM_SLICES);
	}

	public static void compute(int seed, int numVertices, int numEdges, int numSlices) throws Exception
	{
		StarPU.init();
		System.out.println(String.format("TC[seed=%d #vertices=%d #edges=%d]", seed, numVertices, numEdges));

		DataPairSet<Integer,Integer> tc = new DataPairSet<>(generateGraph(seed, numVertices, numEdges), numSlices);

		// Linear transitive closure: each round grows paths by one edge,
		// by joining the graph's edges with the already-discovered paths.
		// e.g. join the path (y, z) from the TC with the edge (x, y) from
		// the graph to obtain the path (x, z).

		// Because join() joins on keys, the edges are stored in reversed order.

		DataPairSet<Integer, Integer> edges = tc.mapToPair(e -> new Tuple2<>(e._2(), e._1()));
		assert(edges.getSize() == tc.getSize());

		long oldCount;
		long nextCount = tc.getSize();

		int round = 1;
		do
		{
			System.out.println("New round "+(round++));
			oldCount = nextCount;
			System.out.println("Old size = "+oldCount);

			System.out.println(" join");
			DataPairSet<Integer, Tuple2<Integer, Integer>> join = tc.join(edges);
			System.out.println(" join size = " + join.getSize());

			System.out.println(" joinpairs");
			DataPairSet<Integer, Integer> joinpairs = join.mapToPair(ProjectFn.INSTANCE);
			System.out.println(" joinpairs size = " + joinpairs.getSize());

			System.out.println(" union");
			DataPairSet<Integer, Integer> union = (DataPairSet<Integer, Integer>) tc.union(joinpairs);
			System.out.println(" union size = " + union.getSize());

			System.out.println(" distinct");
			tc = (DataPairSet<Integer, Integer>) union.distinct().partition(numSlices);
			System.out.println(" distinct size = " + tc.getSize());

			nextCount = tc.getSize();
		} while (nextCount != oldCount);

		//        for(Tuple2<Integer,Integer> t : tc.collect()) {
		//            System.out.println("edge "+ t._1()+ " -> "+ t._2());
		//        }

		System.out.println("TC has " + tc.getSize() + " edges.");

		StarPU.shutdown();
	}
}
