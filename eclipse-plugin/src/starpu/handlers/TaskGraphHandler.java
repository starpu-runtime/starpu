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

import java.awt.EventQueue;
import java.awt.Image;
import java.io.File;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.eclipse.core.commands.AbstractHandler;
import org.eclipse.core.commands.ExecutionEvent;
import org.eclipse.core.commands.ExecutionException;

public class TaskGraphHandler extends AbstractHandler {

	@Override
	public Object execute(ExecutionEvent event) throws ExecutionException {
		EventQueue.invokeLater(() -> {
			try {
				String workDir = System.getProperty("user.dir") + "/" + TraceUtils.getRandomDirectoryName();
				String inputfilename = workDir + "/dag.dot";
				File f = new File(inputfilename);
				if (!f.isFile())
					throw new Exception("File <" + inputfilename + "> does not exist. Have you run StarPU FxT tool?");

				String[] cmd2 = { "dot", "-Tpng", inputfilename, "-o", workDir + "/" + "output.png" };
				starpu.handlers.TraceUtils.runCommand(cmd2);
				String[] cmd3 = { "starpu_tasks_rec_complete", workDir + "/" + "tasks.rec" };
				starpu.handlers.TraceUtils.runCommand(cmd3);

				JFrame frame = new JFrame();
				File imageFile = new File(workDir + "/" + "output.png");
				Image i = ImageIO.read(imageFile);
				ImageIcon image = new ImageIcon(i);
				JLabel imageLabel = new JLabel(image);
				frame.add(imageLabel);
				frame.pack();
				imageLabel.setVisible(true);
				frame.setVisible(true);
				frame.setTitle("StarPU application: Task Graph.png");
			} catch (Exception e) {
				TraceUtils.displayMessage("Error: " + e.toString());
				e.printStackTrace();
			}

		});

		return null;
	}

}
