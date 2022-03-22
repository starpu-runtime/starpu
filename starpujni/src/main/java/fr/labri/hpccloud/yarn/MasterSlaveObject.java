package fr.labri.hpccloud.yarn;

import org.apache.hadoop.fs.CreateFlag;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileContext;
import org.apache.hadoop.fs.Path;

import java.io.IOException;
import java.io.PrintStream;
import java.util.EnumSet;

import static fr.labri.hpccloud.yarn.Constants.*;

public class MasterSlaveObject {
    protected String appPathPrefix;
    protected PrintStream out;
    protected PrintStream err;

    protected PrintStream createAppOutputFile(String suffix) throws IOException {
        Path path = new Path(appPathPrefix, suffix);
        FileContext fc = FileContext.getFileContext();
        FSDataOutputStream out = fc.create(path, EnumSet.of(CreateFlag.CREATE));
        return new PrintStream(out);
    }

    protected MasterSlaveObject(String outputPrefix) throws IOException {
        this();
        setPrefix(outputPrefix);
    }

    protected MasterSlaveObject() throws IOException {
        appPathPrefix = Utils.getMandatoryEnv(ENV_APPLICATION_DIRECTORY);
        out = null;
        err = null;
    }

    protected void setPrefix(String prefix) throws IOException {
        if (out != null || err != null) {
            throw new IllegalStateException("output stream already assigned");
        }
        out = createAppOutputFile(prefix + ".out");
        err = createAppOutputFile(prefix + ".err");
    }

    protected void terminate() {
        out.flush();
        out.close();
        err.flush();
        err.close();
    }

    public static class Master extends MasterSlaveObject {
        protected Master(String prefix) throws IOException {
            super(prefix);
        }
    }

    public static class Slave extends MasterSlaveObject {
        protected String slaveID;
        protected String masterIP;

        protected Slave(String prefix) throws IOException {
            super();
            slaveID = Utils.getMandatoryEnv(ENV_APPLICATION_SLAVE_ID);
            masterIP = Utils.getMandatoryEnv(ENV_APPLICATION_MASTER_HOSTNAME);
            setPrefix(prefix + slaveID);
        }
    }
}
