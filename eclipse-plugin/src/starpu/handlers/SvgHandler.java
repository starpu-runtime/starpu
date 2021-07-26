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
import java.io.PrintWriter;
import java.util.regex.Pattern;

import org.eclipse.core.commands.AbstractHandler;
import org.eclipse.core.commands.ExecutionEvent;
import org.eclipse.core.commands.ExecutionException;
import org.eclipse.core.runtime.IPath;
import org.eclipse.ui.IEditorInput;
import org.eclipse.ui.IPathEditorInput;
import org.eclipse.ui.handlers.HandlerUtil;

public class SvgHandler extends AbstractHandler {

	@Override
	public Object execute(ExecutionEvent event) throws ExecutionException {
		EventQueue.invokeLater(() -> {
			try {
				String workDir = System.getProperty("user.dir") + "/" + TraceUtils.getRandomDirectoryName();
				String inputfilename = workDir + "/dag.dot";
				File f = new File(inputfilename);
				if (!f.isFile())
					throw new Exception("File <" + inputfilename + "> does not exist. Have you run StarPU FxT tool?");

				String[] cmd1 = { "dot", "-Tcmapx", inputfilename, "-o", workDir + "/output.map"};
				TraceUtils.runCommand(cmd1);

				String[] cmd2 = { "dot", "-Tsvg", inputfilename, "-o", workDir + "/output.svg" };
				TraceUtils.runCommand(cmd2);

				IEditorInput input = HandlerUtil.getActiveEditor(event).getEditorInput();

				if (!(input instanceof IPathEditorInput)) {
					System.out.println("There is no path");
				}
				else
				{
					String map = TraceUtils.readFileToString(workDir + "/output.map");
					Pattern p = Pattern.compile("href=\"([^#\"/]+/)*");
					IPath ipath = ((IPathEditorInput) input).getPath().makeAbsolute().removeLastSegments(1);
					String path = ipath.toString();
					String replaceBy = "href=\"" + path + "/";
					map = p.matcher(map).replaceAll(replaceBy);

					PrintWriter pw = new PrintWriter(workDir + "/output.html");
					pw.println(new String("<html>\n" + "<img src=\"output.svg\" usemap=\"#G\" />\n"));
					pw.println(map);
					pw.println(new String("</html>"));
					pw.close();
				}

				String[] cmd8 = { "firefox", workDir + "/output.html" };
				TraceUtils.runCommand(cmd8);
			} catch (Exception e) {
				TraceUtils.displayMessage("Error: " + e.toString());
				e.printStackTrace();
			}

		});

		return null;
	}

}
