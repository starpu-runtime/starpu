package fr.labri.hpccloud.yarn.examples.starpu.nativetests;

import fr.labri.hpccloud.starpu.StarPU;
import fr.labri.hpccloud.starpu.StarPUException;
import fr.labri.hpccloud.yarn.MasterSlaveObject;

import java.io.IOException;

public class StarPUMaster extends MasterSlaveObject.Master {
    public StarPUMaster(String prefix) throws IOException {
        super(prefix);
    }

    public void run() {
        out.println("*** Starting native tests on master");
        if (StarPU.runNativeTests()) {
            out.println("SUCCESS");
        } else {
            out.println("FAIL");
        }
        out.println("*** Ending native tests on master");
    }

    public static void main(String[] args) throws StarPUException, IOException {
        StarPU.init();
        StarPUMaster master = new StarPUMaster("master");
        master.run();
        master.terminate();
        StarPU.shutdown();
        System.exit(0);
    }
}
