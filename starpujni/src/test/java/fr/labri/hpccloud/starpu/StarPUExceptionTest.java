package fr.labri.hpccloud.starpu;

import org.junit.*;

import static org.junit.Assert.*;

public class StarPUExceptionTest {
    private static String message = "message";

    @BeforeClass
    public static void beforeClass() throws Exception {
        StarPU.init();
    }

    @AfterClass
    public static void afterClass() throws Exception {
        StarPU.shutdown();
    }

    @Test(expected = StarPUException.class)
    public void throwStarPUException() throws Exception {
        StarPUException.throwStarPUException(message);
    }

    @Test(expected = StarPUException.class)
    public void throwStarPUExceptionCheckMessage() throws StarPUException {
        try {
            StarPUException.throwStarPUException(message);
        } catch(Exception e) {
            assertEquals(message, e.getMessage());
            throw e;
        }
    }
}