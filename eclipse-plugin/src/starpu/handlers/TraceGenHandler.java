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
import org.eclipse.jface.dialogs.MessageDialog;
import org.eclipse.ui.IWorkbenchWindow;
import org.eclipse.ui.handlers.HandlerUtil;

public class TraceGenHandler extends AbstractHandler {
	@Override
	public Object execute(ExecutionEvent event) throws ExecutionException {
		IWorkbenchWindow window = HandlerUtil.getActiveWorkbenchWindowChecked(event);
		MessageDialog.openInformation(window.getShell(), "StarPU FxT Tool",
				"Running Starpu FxT Tool: generation of different trace formats");
		EventQueue.invokeLater(() -> {
			try {
				String value = System.getenv("STARPU_FXT_PREFIX");
				if (value != null) {
					System.out.println("STARPU_FXT_PREFIX=" + value);
				} else {
					System.out.println("STARPU_FXT_PREFIX does not have a value");
					value = "/tmp";
				}

				String value1 = System.getenv("STARPU_FXT_SUFFIX");
				if (value1 != null) {
					System.out.println("STARPU_FXT_SUFFIX=" + value1);
				} else {
					System.out.println("STARPU_FXT_SUFFIX does not have a value");
					String value2 = System.getenv("USER");
					value1 = "prof_file_" + value2 + "_0";
				}

				String inputfilename = value + "/" + value1;
				File f = new File(inputfilename);
				if (!f.isFile())
					throw new Exception("File <" + inputfilename + "> does not exist. Have you run your application?");

				String[] command = {"starpu_fxt_tool", "-i", inputfilename, "-d", TraceUtils.getRandomDirectoryName(), "-c", "-no-acquire"};
				TraceUtils.runCommand(command);
			} catch (Exception e) {
				TraceUtils.displayMessage("Error: " + e.toString());
				e.printStackTrace();
			}

		});

		return null;
	}

}
