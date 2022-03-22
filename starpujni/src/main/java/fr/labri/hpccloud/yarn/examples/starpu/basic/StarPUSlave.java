package fr.labri.hpccloud.yarn.examples.starpu.basic;

import fr.labri.hpccloud.starpu.StarPU;
import fr.labri.hpccloud.starpu.StarPUException;
import fr.labri.hpccloud.yarn.MasterSlaveObject;

import java.io.IOException;

public class StarPUSlave extends MasterSlaveObject.Slave {
    public StarPUSlave(String prefix) throws IOException {
        super(prefix);
    }

    public void run() {
        out.println("I'm the StarPU slave "+slaveID);
        out.println("I'm in PWD="+System.getenv("PWD"));
    }

    public static void main(String[] args) throws StarPUException, IOException {
        StarPU.init();
        StarPUSlave slave = new StarPUSlave("slave-");
        slave.run();
        slave.terminate();
        StarPU.shutdown();
    }
}
