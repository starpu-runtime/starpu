package fr.labri.hpccloud.starpu.data;

import fr.labri.hpccloud.starpu.StarPUTest;
import org.junit.Test;

import static org.junit.Assert.*;

public class IntegerVariableHandleTest extends StarPUTest {
    @Test
    public void registerUnregisterTest() {
        IntegerVariableHandle var = IntegerVariableHandle.register();
        var.unregister();
    }

    @Test
    public void setGetTest() {
        int checkedValues[] = { 12345, 987654 };
        IntegerVariableHandle var = IntegerVariableHandle.register();
        for (int val: checkedValues) {
            var.setValue(val);
            assertEquals(val, var.getValue());
        }
        var.unregister();
    }
}