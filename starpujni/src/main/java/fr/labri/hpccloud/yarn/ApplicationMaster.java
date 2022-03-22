/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package fr.labri.hpccloud.yarn;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.StringReader;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import org.apache.hadoop.fs.FileSystem;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DataOutputBuffer;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.security.Credentials;
import org.apache.hadoop.security.UserGroupInformation;
import org.apache.hadoop.security.token.Token;
import org.apache.hadoop.util.ExitUtil;
import org.apache.hadoop.util.Shell;
import org.apache.hadoop.yarn.api.ApplicationConstants;
import org.apache.hadoop.yarn.api.ApplicationConstants.Environment;
import org.apache.hadoop.yarn.api.ApplicationMasterProtocol;
import org.apache.hadoop.yarn.api.ContainerManagementProtocol;
import org.apache.hadoop.yarn.api.protocolrecords.AllocateRequest;
import org.apache.hadoop.yarn.api.protocolrecords.AllocateResponse;
import org.apache.hadoop.yarn.api.protocolrecords.FinishApplicationMasterRequest;
import org.apache.hadoop.yarn.api.protocolrecords.RegisterApplicationMasterResponse;
import org.apache.hadoop.yarn.api.protocolrecords.StartContainerRequest;
import org.apache.hadoop.yarn.api.records.*;
import org.apache.hadoop.yarn.client.api.AMRMClient.ContainerRequest;
import org.apache.hadoop.yarn.client.api.async.AMRMClientAsync;
import org.apache.hadoop.yarn.client.api.async.NMClientAsync;
import org.apache.hadoop.yarn.client.api.async.impl.NMClientAsyncImpl;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.hadoop.yarn.security.AMRMTokenIdentifier;
import org.apache.hadoop.yarn.util.ConverterUtils;
import org.apache.log4j.LogManager;

import static fr.labri.hpccloud.yarn.Constants.*;
import static fr.labri.hpccloud.yarn.TimelinePublisher.DSEvent.*;
import static fr.labri.hpccloud.yarn.Utils.getFsApplicationDirectory;

/**
 * An ApplicationMaster for executing shell commands on a set of launched
 * containers using the YARN framework.
 *
 * <p>
 * This class is meant to act as an example on how to write yarn-based
 * application masters.
 * </p>
 *
 * <p>
 * The ApplicationMaster is started on a container by the
 * <code>ResourceManager</code>'s launcher. The first thing that the
 * <code>ApplicationMaster</code> needs to do is to connect and register itself
 * with the <code>ResourceManager</code>. The registration sets up information
 * within the <code>ResourceManager</code> regarding what host:port the
 * ApplicationMaster is listening on to provide any form of functionality to a
 * client as well as a tracking url that a client can use to keep track of
 * status/job history if needed. However, in the distributedshell, trackingurl
 * and appMasterHost:appMasterRpcPort are not supported.
 * </p>
 *
 * <p>
 * The <code>ApplicationMaster</code> needs to send a heartbeat to the
 * <code>ResourceManager</code> at regular intervals to inform the
 * <code>ResourceManager</code> that it is up and alive. The
 * {@link ApplicationMasterProtocol#allocate} to the <code>ResourceManager</code> from the
 * <code>ApplicationMaster</code> acts as a heartbeat.
 *
 * <p>
 * For the actual handling of the job, the <code>ApplicationMaster</code> has to
 * request the <code>ResourceManager</code> via {@link AllocateRequest} for the
 * required no. of containers using {@link ResourceRequest} with the necessary
 * resource specifications such as node location, computational
 * (memory/disk/cpu) resource requirements. The <code>ResourceManager</code>
 * responds with an {@link AllocateResponse} that informs the
 * <code>ApplicationMaster</code> of the set of newly allocated containers,
 * completed containers as well as current state of available resources.
 * </p>
 *
 * <p>
 * For each allocated container, the <code>ApplicationMaster</code> can then set
 * up the necessary launch context via {@link ContainerLaunchContext} to specify
 * the allocated container id, local resources required by the executable, the
 * environment to be setup for the executable, commands to execute, etc. and
 * submit a {@link StartContainerRequest} to the {@link ContainerManagementProtocol} to
 * launch and execute the defined commands on the given allocated container.
 * </p>
 *
 * <p>
 * The <code>ApplicationMaster</code> can monitor the launched container by
 * either querying the <code>ResourceManager</code> using
 * {@link ApplicationMasterProtocol#allocate} to get updates on completed containers or via
 * the {@link ContainerManagementProtocol} by querying for the status of the allocated
 * container's {@link ContainerId}.
 *
 * <p>
 * After the job has been completed, the <code>ApplicationMaster</code> has to
 * send a {@link FinishApplicationMasterRequest} to the
 * <code>ResourceManager</code> to inform it that the
 * <code>ApplicationMaster</code> has been completed.
 */
public class ApplicationMaster {
    public static final Log LOG = LogFactory.getLog(ApplicationMaster.class);

    private static final String[] REQUIRED_ENV_VARIABLES = new String[]{
            ApplicationConstants.APP_SUBMIT_TIME_ENV,
            Environment.CONTAINER_ID.name(),
            Environment.NM_HOST.name(),
            Environment.NM_HTTP_PORT.name(),
            Environment.NM_PORT.name()
    };

    private static final int RM_HEARTBEAT_MS = 1000;

    private static final Options OPTIONS = new Options();

    private static void addOption(String optName, boolean hasArg, String desc) {
        OPTIONS.addOption(null, optName, hasArg, desc);
    }

    static {
        addOption(OPT_NATIVE_LIBDIR, true, "Specify the directory for native libraries");
        addOption(OPT_APPNAME, true, "Name of the application");
        addOption(OPT_APP_ATTEMPT_ID, true, "App Attempt ID. Not to be used unless for testing purposes");
        addOption(OPT_CONTAINER_MASTER_CLASS, true, "Class that implements the master");
        addOption(OPT_CONTAINER_SLAVE_CLASS, true, "Class that implements the slave");
        addOption(OPT_CONTAINER_ENV, true, "Environment for containers. Specified as env_key=env_val pairs");
        addOption(OPT_CONTAINER_MEMORY, true, "Amount of memory in MB to be requested to run the shell command");
        addOption(OPT_CONTAINER_VCORES, true, "Amount of virtual cores to be requested to run the shell command");
        addOption(OPT_NUM_CONTAINERS, true, "No. of containers on which the shell command needs to be executed");
        addOption(OPT_PRIORITY, true, "Application Priority. Default 0");
        addOption(OPT_DEBUG, false, "Dump out debug information");
        addOption(OPT_HELP, false, "Print usage");
    }

    // Configuration
    private Configuration conf;
    private FileSystem fs = null;

    // Handle to communicate with the Resource Manager
    private AMRMClientAsync amRMClient;

    // In both secure and non-secure modes, this points to the job-submitter.
    UserGroupInformation appSubmitterUgi;

    // Handle to communicate with the Node Manager
    private NMClientAsync nmClientAsync;
    private ConcurrentMap<ContainerId, Container> containers = new ConcurrentHashMap<ContainerId, Container>();

    // Application Attempt Id ( combination of attemptId and fail count )
    protected ApplicationAttemptId appAttemptID;

    private String optAppName = null;
    // For status update for clients - yet to be implemented
    // Hostname of the container
    private String appMasterHostname = ""; // NetUtils.getHostname();
    // Port on which the app master listens for status updates from clients
    private int appMasterRpcPort = -1;
    // Tracking url to which app master publishes info for clients to monitor
    private String appMasterTrackingUrl = "";

    // App Master configuration
    // No. of containers to run shell command on
    protected int numTotalContainers = 1;
    // Memory to request for the container on which the shell command will run
    private int containerMemory = 10;
    // VirtualCores to request for the container on which the shell command will run
    private int containerVirtualCores = 1;
    private String optContainerMasterClass = null;
    private String optContainerSlaveClass = null;

    private boolean optDebugFlag = false;

    private String optNativeLibDirectory = null;

    // Priority of the request
    private int requestPriority;

    // Counter for completed containers ( complete denotes successful or failed )
    private AtomicInteger numCompletedContainers = new AtomicInteger();
    // Allocated container count so that we know how many containers has the RM
    // allocated to us

    protected AtomicInteger numAllocatedContainers = new AtomicInteger();
    // Count of failed containers
    private AtomicInteger numFailedContainers = new AtomicInteger();
    // Count of containers already requested from the RM
    // Needed as once requested, we should not request for containers again.
    // Only request for more if the original requirement changes.

    protected AtomicInteger numRequestedContainers = new AtomicInteger();

    private Map<String, String> optContainerEnv = new HashMap<String, String>();

    // Timeline domain ID
    private String domainId = null;

    private volatile boolean done;

    // Launch threads
    private List<Thread> launchThreads = new ArrayList<Thread>();

    // Timeline Client
    private TimelinePublisher timelineClient;

    private NMCallbackHandler nmCallbackHandler = new NMCallbackHandler();
    private RMCallbackHandler rmCallbackHandler = new RMCallbackHandler();


    /**
     * @param args Command line args
     */
    public static void main(String[] args) {
        boolean result = false;
        try {
            ApplicationMaster appMaster = new ApplicationMaster();
            LOG.info("Initializing ApplicationMaster");
            boolean doRun = appMaster.init(args);
            if (!doRun) {
                System.exit(0);
            }
            appMaster.run();
            result = appMaster.finish();
        } catch (Throwable t) {
            LOG.fatal("Error running ApplicationMaster", t);
            LogManager.shutdown();
            ExitUtil.terminate(1, t);
        }
        if (result) {
            LOG.info("Application Master completed successfully. exiting");
            System.exit(0);
        } else {
            LOG.info("Application Master failed. exiting");
            System.exit(2);
        }
    }

    /**
     * Dump out contents of $CWD and the environment to stdout for debugging
     */
    private void dumpOutDebugInfo() {
        if (!optDebugFlag)
            return;

        LOG.info("Dump debug output");
        Map<String, String> envs = System.getenv();
        for (Map.Entry<String, String> env : envs.entrySet()) {
            LOG.info("System env: key=" + env.getKey() + ", val=" + env.getValue());
            System.out.println("System env: key=" + env.getKey() + ", val="
                    + env.getValue());
        }

        BufferedReader buf = null;
        try {
            String lines = Shell.WINDOWS ? Shell.execCommand("cmd", "/c", "dir") :
                    Shell.execCommand("ls", "-al");
            buf = new BufferedReader(new StringReader(lines));
            String line = "";
            while ((line = buf.readLine()) != null) {
                LOG.info("System CWD content: " + line);
                System.out.println("System CWD content: " + line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            IOUtils.cleanup(LOG, buf);
        }
    }

    public ApplicationMaster() {
        // Set up the configuration
        conf = new YarnConfiguration();
    }

    /**
     * Parse command line options
     *
     * @param args Command line args
     * @return Whether init successful and run should be invoked
     * @throws ParseException
     */
    public boolean init(String[] args) throws ParseException {
        CommandLine cliParser = new GnuParser().parse(OPTIONS, args);

        if (args.length == 0) {
            printUsage(OPTIONS);
            throw new IllegalArgumentException("No args specified for application master to initialize");
        }

        //Check whether customer log4j.properties file exists
        if (fileExist(LOCAL_RSRC_LOG4J_PROPERTIES)) {
            try {
                Utils.updateLog4jConfiguration(ApplicationMaster.class, LOCAL_RSRC_LOG4J_PROPERTIES);
            } catch (Exception e) {
                LOG.warn("Can not set up custom log4j properties. " + e);
            }
        }

        if (cliParser.hasOption(OPT_HELP)) {
            printUsage(OPTIONS);
            return false;
        }

        optAppName = Utils.getMandatoryOption(cliParser, OPT_APPNAME);
        optDebugFlag = cliParser.hasOption(OPT_DEBUG);
        dumpOutDebugInfo();

        Map<String, String> envs = Utils.checkEnvironmentVariables(REQUIRED_ENV_VARIABLES);
        ContainerId containerId = ConverterUtils.toContainerId(envs.get(Environment.CONTAINER_ID.name()));
        appAttemptID = containerId.getApplicationAttemptId();

        LOG.info("Application master for app" + ", appId="
                + appAttemptID.getApplicationId().getId() + ", clustertimestamp="
                + appAttemptID.getApplicationId().getClusterTimestamp()
                + ", attemptId=" + appAttemptID.getAttemptId());

        Utils.parseKeyValueOption(cliParser, OPT_CONTAINER_ENV, optContainerEnv);

        requestPriority = Utils.parseIntegerOption(cliParser, OPT_PRIORITY, DEFAULT_PRIORITY);
        containerMemory = Utils.parseIntegerOption(cliParser, OPT_CONTAINER_MEMORY, DEFAULT_CONTAINER_MEMORY);
        containerVirtualCores = Utils.parseIntegerOption(cliParser, OPT_CONTAINER_VCORES, DEFAULT_CONTAINER_VCORES);
        numTotalContainers = Utils.parseIntegerOption(cliParser, OPT_NUM_CONTAINERS, DEFAULT_NUM_CONTAINERS);
        optContainerMasterClass = Utils.getMandatoryOption(cliParser, OPT_CONTAINER_MASTER_CLASS);
        optContainerSlaveClass = Utils.getMandatoryOption(cliParser, OPT_CONTAINER_SLAVE_CLASS);
        optNativeLibDirectory = cliParser.getOptionValue(OPT_NATIVE_LIBDIR, null);

        if (numTotalContainers < 2) {
            throw new IllegalArgumentException("cannot run distributed master/slave app with less than 2 containers");
        }

        return true;
    }

    public String getContainerMasterClass() {
        return optContainerMasterClass;
    }

    public String getContainerSlaveClass() {
        return optContainerSlaveClass;
    }

    public int getContainerMemory() {
        return containerMemory;
    }

    public Map<String, String> getContainerEnv() {
        return optContainerEnv;
    }

    public boolean getDebugFlag() {
        return optDebugFlag;
    }

    /**
     * Helper function to print usage
     *
     * @param opts Parsed command line options
     */
    private void printUsage(Options opts) {
        new HelpFormatter().printHelp("ApplicationMaster", opts);
    }

    /**
     * Main run function for the application master
     *
     * @throws YarnException
     * @throws IOException
     */
    public void run() throws YarnException, IOException, InterruptedException {
        LOG.info("***** Starting ApplicationMaster");

        // Create appSubmitterUgi and add original tokens to it
        String appSubmitterUserName = System.getenv(ApplicationConstants.Environment.USER.name());
        appSubmitterUgi = UserGroupInformation.createRemoteUser(appSubmitterUserName);
        appSubmitterUgi.addCredentials(UserGroupInformation.getCurrentUser().getCredentials());

        // Client between the Application Master and the Resource Manager. Parameters are:
        // - specify heartbeat period
        // - callbacks invoked when RM events occur
        // Heartbeat is started only when the AM has been registered to the RM using registerApplicationMaster method.
        amRMClient = AMRMClientAsync.createAMRMClientAsync(RM_HEARTBEAT_MS, rmCallbackHandler);
        amRMClient.init(conf);
        amRMClient.start();

        // Client between Application Master with all Node Managers. Parameters are:
        // - callback invoked when events on containers occur.
        nmClientAsync = new NMClientAsyncImpl(nmCallbackHandler);
        nmClientAsync.init(conf);
        nmClientAsync.start();

        // Client between Application Master and the Timeline server.
        timelineClient = TimelinePublisher.startTimelineClient(appSubmitterUgi, conf);
        timelineClient.publishApplicationAttemptEvent(appAttemptID.toString(), DS_APP_ATTEMPT_START, domainId);

        // Register self with ResourceManager
        // This will start heartbeating to the RM
        RegisterApplicationMasterResponse response =
                amRMClient.registerApplicationMaster(appMasterHostname, appMasterRpcPort, appMasterTrackingUrl);
        // Dump out information about cluster capability as seen by the resource manager
        int maxMem = response.getMaximumResourceCapability().getMemory();
        int maxVCores = response.getMaximumResourceCapability().getVirtualCores();
        List<Container> previousAMRunningContainers = response.getContainersFromPreviousAttempts();

        LOG.info("Max mem capabililty of resources in this cluster " + maxMem);
        LOG.info("Max vcores capabililty of resources in this cluster " + maxVCores);

        if (containerMemory > maxMem) {
            LOG.info("Container memory specified above max threshold of cluster."
                    + " Using max value." + ", specified=" + containerMemory + ", max="
                    + maxMem);
            containerMemory = maxMem;
        }

        if (containerVirtualCores > maxVCores) {
            LOG.info("Container virtual cores specified above max threshold of cluster."
                    + " Using max value." + ", specified=" + containerVirtualCores + ", max="
                    + maxVCores);
            containerVirtualCores = maxVCores;
        }

        LOG.info(String.format("%s received %d previous attempts' running containers on AM registration.",
                appAttemptID, previousAMRunningContainers.size()));
        numAllocatedContainers.addAndGet(previousAMRunningContainers.size());

        int numTotalContainersToRequest = numTotalContainers - previousAMRunningContainers.size();
        for (int i = 0; i < numTotalContainersToRequest; ++i) {
            ContainerRequest request = setupContainerAskForRM();
            amRMClient.addContainerRequest(request);
        }
        numRequestedContainers.set(numTotalContainers);
    }

    private ContainerRequest setupContainerAskForRM() {
        Priority pri = Priority.newInstance(requestPriority);
        Resource capability = Resource.newInstance(containerMemory, containerVirtualCores);
        ContainerRequest request = new ContainerRequest(capability, null, null, pri);
        LOG.info("Requested container ask: " + request.toString());

        return request;
    }

    protected boolean finish() {
        // wait for completion.
        while (!done && (numCompletedContainers.get() != numTotalContainers)) {
            try {
                Thread.sleep(200);
            } catch (InterruptedException ex) {
            }
        }

        timelineClient.publishApplicationAttemptEvent(appAttemptID.toString(), DS_APP_ATTEMPT_END, domainId);

        // Join all launched threads
        // needed for when we time out
        // and we need to release containers
        for (Thread launchThread : launchThreads) {
            try {
                launchThread.join(10000);
            } catch (InterruptedException e) {
                LOG.info("Exception thrown in thread join: " + e.getMessage());
                e.printStackTrace();
            }
        }

        // When the application completes, it should stop all running containers
        LOG.info("Application completed. Stopping running containers");
        nmClientAsync.stop();

        // When the application completes, it should send a finish application
        // signal to the RM
        LOG.info("Application completed. Signalling finish to RM");

        FinalApplicationStatus appStatus;
        String appMessage = null;
        boolean success = numFailedContainers.get() == 0 && numCompletedContainers.get() == numTotalContainers;
        if (success) {
            appStatus = FinalApplicationStatus.SUCCEEDED;
        } else {
            appStatus = FinalApplicationStatus.FAILED;
            appMessage = "Diagnostics." + ", total=" + numTotalContainers
                    + ", completed=" + numCompletedContainers.get() + ", allocated="
                    + numAllocatedContainers.get() + ", failed="
                    + numFailedContainers.get();
            LOG.info(appMessage);
        }

        try {
            amRMClient.unregisterApplicationMaster(appStatus, appMessage, null);
        } catch (YarnException | IOException e) {
            LOG.error("Failed to unregister application", e);
        }

        amRMClient.stop();
        timelineClient.stop();

        return success;
    }

    public FileSystem getFS() throws IOException {
        if (fs == null)
            fs = FileSystem.get(conf);
        return fs;
    }

    public ApplicationId getAppID() {
        return appAttemptID.getApplicationId();
    }

    public UserGroupInformation getUGI() {
        return appSubmitterUgi;
    }

    public String getAppName() {
        return optAppName;
    }

    public Configuration getConf() {
        return conf;
    }

    public NMClientAsync getNmClient() {
        return nmClientAsync;
    }

    public void countCompletedCointainer() {
        numCompletedContainers.incrementAndGet();
    }

    public void countFailedCointainer() {
        numFailedContainers.incrementAndGet();
    }

    private Map<String, String> buildContainerEnvironment() throws IOException {
        // Set the env variables to be setup in the env where the application master will be run
        LOG.info("Set the environment for the application master");
        Map<String, String> env = new HashMap<>();
        String cp = System.getenv("CLASSPATH");

        StringBuilder classPathEnv = new StringBuilder(ApplicationConstants.Environment.CLASSPATH.$$());
        classPathEnv.append(ApplicationConstants.CLASS_PATH_SEPARATOR).append(cp);
        classPathEnv.append(ApplicationConstants.CLASS_PATH_SEPARATOR).append("./*");
        classPathEnv.append(ApplicationConstants.CLASS_PATH_SEPARATOR).append("./log4j.properties");
        env.put("CLASSPATH", classPathEnv.toString());

        env.put(ENV_APPLICATION_DIRECTORY, getFsApplicationDirectory(getAppName(), getFS(), getAppID()));
        if (optNativeLibDirectory != null) {
            env.put(Environment.LD_LIBRARY_PATH.name(), Environment.LD_LIBRARY_PATH.$$()+":"+ Path.CUR_DIR);
        }
        env.putAll(getContainerEnv());

        return env;
    }

    private List<String> buildContainerCommands(boolean isMaster) {
        LOG.info("Setting up app master command");

        String option = "--%s %s";
        ArrayList<String> args = new ArrayList<>();

        if (getDebugFlag()) {
            args.add("--debug");
        }
        String prefix;
        String mainClass;
        if (isMaster) {
            prefix = "master";
            mainClass = getContainerMasterClass();
        } else {
            prefix = "slave";
            mainClass = getContainerSlaveClass();
        }
        String command = Utils.createJavaCommand(getContainerMemory(), mainClass, prefix, args);

        LOG.info("Completed setting up app master command " + command);
        List<String> result = new ArrayList<>();
        result.add(command);

        return result;
    }

    private static Map<String, LocalResource> localResources = null;

    private Map<String, LocalResource> buildContainerLocalResources(ApplicationId appId, FileSystem fs) throws IOException {
        if (localResources != null) {
            return localResources;
        }

        localResources = new HashMap<String, LocalResource>();

        LOG.info("Copy App Master jar from local filesystem and add to local environment");
        // Copy the application master jar to the filesystem
        // Create a local resource to point to the destination jar path
        Utils.addPathToLocalResources(fs,
                Utils.getPathInAppDir(fs, getAppName(), getAppID(), LOCAL_RSRC_APPLICATION_JAR),
                localResources);
        if(optNativeLibDirectory != null) {
            Utils.addDirToLocalResources(fs, optNativeLibDirectory, localResources);
        }

        return localResources;
    }


    private ContainerLaunchContext getMasterContext() throws IOException {
        Map<String, LocalResource> localResources = buildContainerLocalResources(getAppID(), getFS());
        List<String> masterCommands = buildContainerCommands(true);
        ByteBuffer tokens = getAuthenticationTokens();
        Map<String, String> env = buildContainerEnvironment();

        return ContainerLaunchContext.newInstance(localResources, env, masterCommands, null,
                tokens.duplicate(), null);
    }

    private ContainerLaunchContext getSlaveContext(int slaveID, String masterAddress) throws IOException {
        Map<String, LocalResource> localResources = buildContainerLocalResources(getAppID(), getFS());
        List<String> slaveCommands = buildContainerCommands(false);
        ByteBuffer tokens = getAuthenticationTokens();
        Map<String, String> env = buildContainerEnvironment();
        env.put(ENV_APPLICATION_SLAVE_ID, Integer.toString(slaveID));
        env.put(ENV_APPLICATION_MASTER_HOSTNAME, masterAddress);

        return ContainerLaunchContext.newInstance(localResources, env, slaveCommands, null,
                tokens.duplicate(), null);
    }

    private boolean fileExist(String filePath) {
        return new File(filePath).exists();
    }

    private String readContent(String filePath) throws IOException {
        DataInputStream ds = null;
        try {
            ds = new DataInputStream(new FileInputStream(filePath));
            return ds.readUTF();
        } finally {
            org.apache.commons.io.IOUtils.closeQuietly(ds);
        }
    }

    public ByteBuffer getAuthenticationTokens() throws IOException {
        // Note: Credentials, Token, UserGroupInformation, DataOutputBuffer class
        // are marked as LimitedPrivate
        Credentials credentials = getUGI().getCredentials();
        DataOutputBuffer dob = new DataOutputBuffer();
        credentials.writeTokenStorageToStream(dob);
        // Now remove the AM->RM token so that containers cannot access it.
        Iterator<Token<?>> iter = credentials.getAllTokens().iterator();
        LOG.info("Executing with tokens:");
        while (iter.hasNext()) {
            Token<?> token = iter.next();
            LOG.info(token);
            if (token.getKind().equals(AMRMTokenIdentifier.KIND_NAME)) {
                iter.remove();
            }
        }
        return ByteBuffer.wrap(dob.getData(), 0, dob.getLength());
    }

    private class RMCallbackHandler implements AMRMClientAsync.CallbackHandler {
        AtomicReference<String> masterHostname = new AtomicReference<>(null);
        AtomicInteger slaveID = new AtomicInteger(1);

        @Override
        public void onContainersCompleted(List<ContainerStatus> completedContainers) {
            logInfoContainerStatus(completedContainers);
            for (ContainerStatus containerStatus : completedContainers) {
                // non complete containers should not be here
                assert (containerStatus.getState() == ContainerState.COMPLETE);

                // increment counters for completed/failed containers
                int exitStatus = containerStatus.getExitStatus();
                if (0 != exitStatus) {
                    // container failed
                    if (ContainerExitStatus.ABORTED != exitStatus) {
                        // shell script failed
                        // counts as completed
                        numCompletedContainers.incrementAndGet();
                        numFailedContainers.incrementAndGet();
                    } else {
                        // container was killed by framework, possibly preempted
                        // we should re-try as the container was lost for some reason
                        numAllocatedContainers.decrementAndGet();
                        numRequestedContainers.decrementAndGet();
                        // we do not need to release the container as it would be done
                        // by the RM
                    }
                } else {
                    // nothing to do
                    // container completed successfully
                    numCompletedContainers.incrementAndGet();
                    LOG.info("Container completed successfully." + ", containerId="
                            + containerStatus.getContainerId());
                }
                timelineClient.publishContainerEndEvent(containerStatus, domainId);
            }

            // ask for more containers if any failed
            int askCount = numTotalContainers - numRequestedContainers.get();
            numRequestedContainers.addAndGet(askCount);

            if (askCount > 0) {
                for (int i = 0; i < askCount; ++i) {
                    ContainerRequest containerAsk = setupContainerAskForRM();
                    amRMClient.addContainerRequest(containerAsk);
                }
            }

            if (numCompletedContainers.get() == numTotalContainers) {
                done = true;
            }
        }


        @Override
        public void onContainersAllocated(List<Container> allocatedContainers) {
            logInfoAllocatedContainers(allocatedContainers);

            numAllocatedContainers.addAndGet(allocatedContainers.size());
            for (Container allocatedContainer : allocatedContainers) {
                try {
                    ContainerLaunchContext ctx;
                    if (masterHostname.compareAndSet(null, allocatedContainer.getNodeId().getHost())) {
                        ctx = getMasterContext();
                    } else {
                        ctx = getSlaveContext(slaveID.getAndIncrement(), masterHostname.get());
                    }

                    Thread launchThread =
                            new LaunchContainerThread(allocatedContainer, ApplicationMaster.this, ctx);
                    launchThreads.add(launchThread);
                    launchThread.start();
                } catch (IOException e) {
                    LOG.error("can't create launch context", e);
                    amRMClient.releaseAssignedContainer(allocatedContainer.getId());
                    numFailedContainers.incrementAndGet();
                    numCompletedContainers.incrementAndGet();
                }
            }
        }

        @Override
        public void onShutdownRequest() {
            done = true;
        }

        @Override
        public void onNodesUpdated(List<NodeReport> updatedNodes) {
        }

        @Override
        public float getProgress() {
            // set progress to deliver to RM on next heartbeat
            float progress = (float) numCompletedContainers.get() / numTotalContainers;
            return progress;
        }

        @Override
        public void onError(Throwable e) {
            done = true;
            amRMClient.stop();
        }

        private void logInfoContainerStatus(List<ContainerStatus> completedContainers) {
            LOG.info("Got response from RM for container ask, completedCnt=" + completedContainers.size());
            for (ContainerStatus containerStatus : completedContainers) {
                LOG.info(appAttemptID + " got container status for containerID="
                        + containerStatus.getContainerId() + ", state="
                        + containerStatus.getState() + ", exitStatus="
                        + containerStatus.getExitStatus() + ", diagnostics="
                        + containerStatus.getDiagnostics());
            }
        }

        private void logInfoAllocatedContainers(List<Container> allocatedContainers) {
            LOG.info("Got response from RM for container ask, allocatedCnt=" + allocatedContainers.size());
            for (Container allocatedContainer : allocatedContainers) {
                LOG.info("Launching shell command on a new container."
                        + ", containerId=" + allocatedContainer.getId()
                        + ", containerNode=" + allocatedContainer.getNodeId().getHost()
                        + ":" + allocatedContainer.getNodeId().getPort()
                        + ", containerNodeURI=" + allocatedContainer.getNodeHttpAddress()
                        + ", containerResourceMemory"
                        + allocatedContainer.getResource().getMemory()
                        + ", containerResourceVirtualCores"
                        + allocatedContainer.getResource().getVirtualCores());
            }
        }
    }

    public void addContainer(ContainerId containerId, Container container) {
        containers.putIfAbsent(containerId, container);
    }

    private class NMCallbackHandler implements NMClientAsync.CallbackHandler {
        @Override
        public void onContainerStopped(ContainerId containerId) {
            if (LOG.isDebugEnabled()) {
                LOG.debug("Succeeded to stop Container " + containerId);
            }
            containers.remove(containerId);
        }

        @Override
        public void onContainerStatusReceived(ContainerId containerId, ContainerStatus containerStatus) {
            if (LOG.isDebugEnabled()) {
                LOG.debug("Container Status: id=" + containerId + ", status=" + containerStatus);
            }
        }

        @Override
        public void onContainerStarted(ContainerId containerId, Map<String, ByteBuffer> allServiceResponse) {
            if (LOG.isDebugEnabled()) {
                LOG.debug("Succeeded to start Container " + containerId);
            }
            Container container = containers.get(containerId);
            if (container != null) {
                nmClientAsync.getContainerStatusAsync(containerId, container.getNodeId());
            }
            timelineClient.publishContainerStartEvent(container, domainId);
        }

        @Override
        public void onStartContainerError(ContainerId containerId, Throwable t) {
            LOG.error("Failed to start Container " + containerId);
            containers.remove(containerId);
            numCompletedContainers.incrementAndGet();
            numFailedContainers.incrementAndGet();
        }

        @Override
        public void onGetContainerStatusError(ContainerId containerId, Throwable t) {
            LOG.error("Failed to query the status of Container " + containerId);
        }

        @Override
        public void onStopContainerError(ContainerId containerId, Throwable t) {
            LOG.error("Failed to stop Container " + containerId);
            containers.remove(containerId);
        }
    }
}
