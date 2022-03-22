package fr.labri.hpccloud.starpu.data;

import fr.labri.hpccloud.starpu.StarPUTest;

public abstract class VectorHandleTest extends StarPUTest {

    protected abstract ScalarVectorHandle register(int size);

    protected void getSizeTest() {
        int checkSizes[] = { 1, 2, 5, 1000001 };
        for (int sz : checkSizes ) {
            ScalarVectorHandle v = register(sz);
            assertEquals(sz, v.getSize());
            v.unregister();
        }
    }
}