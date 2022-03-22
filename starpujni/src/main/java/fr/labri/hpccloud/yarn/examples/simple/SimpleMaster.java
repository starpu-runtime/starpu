package fr.labri.hpccloud.yarn.examples.simple;

import fr.labri.hpccloud.yarn.MasterSlaveObject;

import java.io.IOException;

public class SimpleMaster extends MasterSlaveObject.Master {
    public SimpleMaster(String prefix) throws IOException {
        super(prefix);
    }

    public void run() {
        out.println("I'm the Master");
        terminate();
    }

    public static void main(String[] args) throws IOException {
        SimpleMaster master = new SimpleMaster("master");
        master.run();
    }
}
