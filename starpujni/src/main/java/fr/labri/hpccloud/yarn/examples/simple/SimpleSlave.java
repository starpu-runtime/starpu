package fr.labri.hpccloud.yarn.examples.simple;

import fr.labri.hpccloud.yarn.MasterSlaveObject;

import java.io.IOException;

public class SimpleSlave extends MasterSlaveObject.Slave {
    public SimpleSlave(String prefix) throws IOException {
        super(prefix);
    }

    public void run() {
        out.println("I'm slave with ID=" + slaveID );
        out.println("MASTER HOST=" + masterIP);
        terminate();
    }

    public static void main(String[] args) throws IOException  {
        SimpleSlave slave = new SimpleSlave("slave-");
        slave.run();
    }
}
