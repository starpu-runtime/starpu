// StarPU --- Runtime system for heterogeneous multicore architectures.
//
// Copyright (C) 2021       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
package starpu.handlers;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Random;

import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

public class TraceUtils {

	private static int x = 1000 + new Random().nextInt(9999);

	public static void runCommand(String[] command) throws Exception
	{
		System.out.println("Running command " + Arrays.toString(command));
		Process p = Runtime.getRuntime().exec(command);

		String line;
		BufferedReader in = new BufferedReader(new InputStreamReader(p.getInputStream()));
		while ((line = in.readLine()) != null) {
			System.out.println(line);
		}
		in.close();
	}

	public static String getRandomDirectoryName()
	{
		return "traces_" + x;
	}

	public static void displayMessage(String message)
	{
		final JFrame f = new JFrame("StarPU Message");

		JLabel l = new JLabel(message);
		JButton b19 = new JButton("OK");

		b19.addActionListener(new ActionListener()
			{
				public void actionPerformed(ActionEvent evt)
				{
					f.setVisible(false);
				}
			});

		JPanel p = new JPanel();
		p.setLayout(new BoxLayout(p, BoxLayout.Y_AXIS));
		p.add(l);
		p.add(b19);

		f.add(p);
		f.pack();
		f.setVisible(true);
	}

	public static String readFileToString(String filename) throws IOException
	{
		BufferedReader reader = new BufferedReader(new FileReader(filename));
		StringBuilder stringBuilder = new StringBuilder();
		char[] buffer = new char[10];
		while (reader.read(buffer) != -1) {
			stringBuilder.append(new String(buffer));
			buffer = new char[10];
		}
		reader.close();

		return stringBuilder.toString();
	}


}
