package fr.labri.hpccloud.starpu.data;

import fr.labri.hpccloud.starpu.StarPU;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.*;
import java.util.Random;

import static org.junit.Assert.assertEquals;

@RunWith(JUnit4.class)
public class VectorHandleSerializationTest {
    @BeforeClass
    public static void beforeClass() throws Exception {
        StarPU.init();
    }

    @AfterClass
    public static void afterClass() throws Exception {
        StarPU.shutdown();
    }

    private <T> void checkVectorHandle(VectorHandle<T> v1, VectorHandle<T> v2) {
        v1.acquire(DataHandle.AccessMode.STARPU_R);
        v2.acquire(DataHandle.AccessMode.STARPU_R);
        int size = v1.getSize();
        assertEquals(size, v2.getSize());
        for(int i = 0; i < size; i++) {
            assertEquals(v1.getValueAt(i), v2.getValueAt(i));
        }
        v1.release();
        v2.release();
    }

    public <T> void checkSerialization(VectorHandle<T> vector) throws IOException, ClassNotFoundException {
        byte[] out = vector.pack();
        VectorHandle<T> vectorClone = VectorHandle.register(vector.getSize());
        vectorClone.acquire(DataHandle.AccessMode.STARPU_W);
        vectorClone.unpack(out);
        vectorClone.release();
        checkVectorHandle(vector, vectorClone);
        vectorClone.unregister();
    }

    @Test
    public void integerVectorSerializationTest() throws IOException, ClassNotFoundException {
        Random rnd = new Random();
        int checkedSizes[] = {1, 2, 3, 10, 11, 111 };
        int nbTestsPerSize = 100;

        for(int sz : checkedSizes) {
            for (int i = 0 ; i < nbTestsPerSize; i++) {
                VectorHandle<Integer> v = VectorHandle.register(sz);
                v.acquire(DataHandle.AccessMode.STARPU_W);
                for (int j = 0; j < sz; j++)
                    v.setValueAt(j, rnd.nextInt());
                v.release();
                checkSerialization(v);
                v.unregister();
            }
        }
    }
}