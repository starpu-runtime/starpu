package fr.labri.hpccloud.yarn.examples.starpu.nativetests;

import fr.labri.hpccloud.yarn.MasterSlaveObject;


import fr.labri.hpccloud.starpu.StarPU;
import fr.labri.hpccloud.starpu.StarPUException;
import fr.labri.hpccloud.yarn.MasterSlaveObject;

import java.io.IOException;

public class StarPUSlave extends MasterSlaveObject.Slave {
    public StarPUSlave(String prefix) throws IOException {
        super(prefix);
    }

    public void run() {
        System.out.println("*** Starting native tests on slave "+slaveID);
        if (StarPU.runNativeTests()) {
            System.out.println("SUCCESS");
        } else {
            System.out.println("FAIL");
        }
        System.out.println("*** Ending native tests on slave "+slaveID);

        out.println("I'm the StarPU slave "+slaveID);
    }

    public static void main(String[] args) throws StarPUException, IOException {
        StarPU.init();
        StarPUSlave slave = new StarPUSlave("slave-");
        slave.run();
        slave.terminate();
        StarPU.shutdown();
        System.exit(0);
    }
}
