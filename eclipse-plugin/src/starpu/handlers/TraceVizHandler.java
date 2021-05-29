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
import java.io.File;

import org.eclipse.core.commands.AbstractHandler;
import org.eclipse.core.commands.ExecutionEvent;
import org.eclipse.core.commands.ExecutionException;

public class TraceVizHandler extends AbstractHandler {

	@Override
	public Object execute(ExecutionEvent event) throws ExecutionException {
		EventQueue.invokeLater(() -> {
			try {
				String workDir = System.getProperty("user.dir") + "/" + TraceUtils.getRandomDirectoryName();
				String inputfilename = workDir + "/paje.trace";

				File f = new File(inputfilename);
				if (!f.isFile())
					throw new Exception("File <" + inputfilename + "> does not exist. Have you run StarPU FxT tool?");

				String[] cmd1 = { "vite", inputfilename };
				starpu.handlers.TraceUtils.runCommand(cmd1);
			} catch (Exception e) {
				TraceUtils.displayMessage("Error: " + e.toString());
				e.printStackTrace();
			}
		});

		return null;
	}

}
