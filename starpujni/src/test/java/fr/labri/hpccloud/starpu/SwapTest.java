package fr.labri.hpccloud.starpu;

import fr.labri.hpccloud.starpu.StarPU;
import fr.labri.hpccloud.starpu.data.VectorHandle;
import fr.labri.hpccloud.yarn.MasterSlaveObject;
import junit.framework.TestCase;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import static fr.labri.hpccloud.starpu.data.DataHandle.AccessMode.STARPU_R;
import static fr.labri.hpccloud.starpu.data.DataHandle.AccessMode.STARPU_W;

@RunWith(JUnit4.class)
public class SwapTest extends TestCase {
    public static final int VECTOR_SIZE = 1000;
    public static final int NB_REREAD = 10;

    @BeforeClass
    public static void beforeClass() throws Exception {
        StarPU.enableTrace = true;
        VectorHandle.enableTrace = true;
        Map<String, String> env = new HashMap<>();
        String tmpdir = System.getProperty("java.io.tmpdir");
        if (null == tmpdir || tmpdir.length() == 0)
            tmpdir = ".";
        String swapdir = tmpdir + File.separator + "STARPU_SWAP";
        env.put("STARPU_DISK_SWAP", swapdir);
        env.put("STARPU_DISK_SWAP_BACKEND", "unistd");
        env.put("STARPU_DISK_SWAP_SIZE", "200");
        env.put("STARPU_LIMIT_CPU_MEM", "1");

        StarPU.init(env);
    }

    @AfterClass
    public static void afterClass() throws Exception {
        StarPU.shutdown();
    }

    private int updateCheckSum(int prev, int newval) {
        return prev ^ String.valueOf(19 * prev + 111 * newval).hashCode();
    }

    @Test
    public void swapTest() throws Exception {
        Random rnd = new Random();
        ArrayList<VectorHandle<Integer>> vectors = new ArrayList<>();
        int memLimit = 1000;
        int checkSum1 = 0;
        for (int i = 0; i < memLimit; i++) {
            VectorHandle<Integer> v = VectorHandle.register(VECTOR_SIZE);
            vectors.add(v);
            v.acquire(STARPU_W);
            for (int j = 0; j < VECTOR_SIZE; j++) {
                int r = rnd.nextInt();
                checkSum1 = updateCheckSum(checkSum1, r);
                v.setValueAt(j, r);
            }
            v.release();

            for (int j = 0; j < NB_REREAD; j++) {
                VectorHandle<Integer> w = vectors.get(rnd.nextInt(vectors.size()));
                w.acquire(STARPU_R);
                for (int k = 0; k < VECTOR_SIZE; k++) {
                    w.getValueAt(j);
                }
                w.release();
            }
        }
        int checkSum2 = 0;

        for (VectorHandle<Integer> v : vectors) {
            v.acquire(STARPU_R);
            int size = v.getSize();
            for (int j = 0; j < size; j++) {
                int r = v.getValueAt(j);
                checkSum2 = updateCheckSum(checkSum2, r);
            }
            v.release();
            v.unregister();
        }
        assertEquals(checkSum1, checkSum2);
    }
}
