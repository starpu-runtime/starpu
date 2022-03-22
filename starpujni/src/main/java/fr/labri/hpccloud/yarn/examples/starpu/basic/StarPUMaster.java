package fr.labri.hpccloud.yarn.examples.starpu.basic;

import fr.labri.hpccloud.starpu.StarPU;
import fr.labri.hpccloud.starpu.StarPUException;
import fr.labri.hpccloud.yarn.MasterSlaveObject;

import java.io.IOException;

public class StarPUMaster extends MasterSlaveObject.Master {
    public StarPUMaster(String prefix) throws IOException {
        super(prefix);
    }

    public void run() {
        out.println("I'm the StarPU master");
        out.println("I'm in PWD="+System.getenv("PWD"));
    }

    public static void main(String[] args) throws StarPUException, IOException {
        StarPU.init();
        StarPUMaster master = new StarPUMaster("master");
        master.run();
        master.terminate();
        StarPU.shutdown();
    }
}
