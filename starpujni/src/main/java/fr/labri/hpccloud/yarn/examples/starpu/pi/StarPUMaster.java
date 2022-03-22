package fr.labri.hpccloud.yarn.examples.starpu.pi;

import fr.labri.hpccloud.starpu.StarPU;
import fr.labri.hpccloud.starpu.StarPUException;
import fr.labri.hpccloud.starpu.examples.Pi3;
import fr.labri.hpccloud.yarn.MasterSlaveObject;
import fr.labri.hpccloud.yarn.Utils;

import java.io.IOException;

public class StarPUMaster extends MasterSlaveObject.Master {
    public StarPUMaster(String prefix) throws IOException {
        super(prefix);
    }

    public void run() throws StarPUException, IOException {
        out.println("I'm the StarPU master");
        out.println("I'm in PWD="+System.getenv("PWD"));
        int nbSlices = Integer.parseInt(Utils.getMandatoryEnv("NB_SLICES"));
        Pi3.compute(out, nbSlices);
    }

    public static void main(String[] args) throws StarPUException, IOException {
        StarPUMaster master = new StarPUMaster("pi-master");
        master.run();
        master.terminate();
    }
}
