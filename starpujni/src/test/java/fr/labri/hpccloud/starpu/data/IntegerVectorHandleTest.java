package fr.labri.hpccloud.starpu.data;

import org.junit.Test;

import java.util.Random;

public class IntegerVectorHandleTest extends VectorHandleTest {
    protected ScalarVectorHandle register(int size) {
        return IntegerVectorHandle.register(size);
    }

    @Override
    @Test
    public void getSizeTest() {
        super.getSizeTest();
    }

    @Test
    public void setGetTest() {
        Random rnd = new Random();
        int checkedSizes[] = {1, 3, 111 };
        for(int sz : checkedSizes) {
            IntegerVectorHandle hdl = (IntegerVectorHandle) register(sz);
            for (int i = 0 ; i < sz; i++) {
                int value = rnd.nextInt();
                hdl.setValueAt(i, value);
                assertEquals(value, hdl.getValueAt(i));
            }
            hdl.unregister();
        }
    }
}