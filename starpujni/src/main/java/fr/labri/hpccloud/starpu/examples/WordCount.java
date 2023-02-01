//! [To be included. You should update doxygen if you see this text.]
package fr.labri.hpccloud.starpu.examples;

import fr.labri.hpccloud.starpu.StarPU;
import fr.labri.hpccloud.starpu.data.DataPairSet;
import fr.labri.hpccloud.starpu.data.DataSet;
import fr.labri.hpccloud.starpu.data.Tuple2;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.Arrays;
import java.util.regex.Pattern;

public class WordCount
{
	static InputStream openFile(String filename) throws Exception
	{
		return WordCount.class.getResourceAsStream(filename);
	}

	private static final Pattern SPACE = Pattern.compile(" ");

	public static void main(String[] args ) throws Exception
	{
		InputStream input = new FileInputStream(args[0]);
		StarPU.init();
		compute(input);
		input.close();
		StarPU.shutdown();
	}

	private static void compute(InputStream input) throws Exception
	{
		DataSet<String> lines = DataSet.readFile (input, s->s).splitByBlocks(10);
		DataSet<String> words = lines.flatMap(s -> Arrays.asList(SPACE.split(s)).iterator()).splitByBlocks(10);
		DataPairSet<String,Integer> ones = (DataPairSet<String,Integer>)words.mapToPair(w-> new Tuple2<>(w,1));
		DataPairSet<String,Integer> counts = ones.reduceByKey((c1,c2)-> c1 + c2);

		for(Tuple2<String,Integer> p : counts.collect())
		{
			System.out.println("("+p._1()+","+p._2()+")");
		}
	}
}
//! [To be included. You should update doxygen if you see this text.]
