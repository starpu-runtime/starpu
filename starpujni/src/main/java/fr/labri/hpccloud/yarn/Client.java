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

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.DataOutputBuffer;
import org.apache.hadoop.security.Credentials;
import org.apache.hadoop.security.UserGroupInformation;
import org.apache.hadoop.security.token.Token;
import org.apache.hadoop.yarn.api.ApplicationClientProtocol;
import org.apache.hadoop.yarn.api.ApplicationConstants;
import org.apache.hadoop.yarn.api.ApplicationConstants.Environment;
import org.apache.hadoop.yarn.api.protocolrecords.GetNewApplicationResponse;
import org.apache.hadoop.yarn.api.protocolrecords.KillApplicationRequest;
import org.apache.hadoop.yarn.api.records.ApplicationId;
import org.apache.hadoop.yarn.api.records.ApplicationReport;
import org.apache.hadoop.yarn.api.records.ApplicationSubmissionContext;
import org.apache.hadoop.yarn.api.records.ContainerLaunchContext;
import org.apache.hadoop.yarn.api.records.FinalApplicationStatus;
import org.apache.hadoop.yarn.api.records.LocalResource;
import org.apache.hadoop.yarn.api.records.NodeReport;
import org.apache.hadoop.yarn.api.records.NodeState;
import org.apache.hadoop.yarn.api.records.Priority;
import org.apache.hadoop.yarn.api.records.QueueACL;
import org.apache.hadoop.yarn.api.records.QueueInfo;
import org.apache.hadoop.yarn.api.records.QueueUserACLInfo;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.api.records.YarnApplicationState;
import org.apache.hadoop.yarn.api.records.YarnClusterMetrics;
import org.apache.hadoop.yarn.api.records.timeline.TimelineDomain;
import org.apache.hadoop.yarn.client.api.TimelineClient;
import org.apache.hadoop.yarn.client.api.YarnClient;
import org.apache.hadoop.yarn.client.api.YarnClientApplication;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.hadoop.yarn.util.timeline.TimelineUtils;

import static fr.labri.hpccloud.yarn.Constants.*;

/**
 * Client for Distributed Shell application submission to YARN.
 *
 * <p> The distributed shell client allows an application master to be launched that in turn would run
 * the provided shell command on a set of containers. </p>
 *
 * <p>This client is meant to act as an example on how to write yarn-based applications. </p>
 *
 * <p> To submit an application, a client first needs to connect to the <code>ResourceManager</code>
 * aka ApplicationsManager or ASM via the {@link ApplicationClientProtocol}. The {@link ApplicationClientProtocol}
 * provides a way for the client to get access to cluster information and to request for a
 * new {@link ApplicationId}. <p>
 *
 * <p> For the actual job submission, the client first has to create an {@link ApplicationSubmissionContext}.
 * The {@link ApplicationSubmissionContext} defines the application details such as {@link ApplicationId}
 * and application name, the priority assigned to the application and the queue
 * to which this application needs to be assigned. In addition to this, the {@link ApplicationSubmissionContext}
 * also defines the {@link ContainerLaunchContext} which describes the <code>Container</code> with which
 * the {@link ApplicationMaster} is launched. </p>
 *
 * <p> The {@link ContainerLaunchContext} in this scenario defines the resources to be allocated for the
 * {@link ApplicationMaster}'s container, the local resources (jars, configuration files) to be made available
 * and the environment to be set for the {@link ApplicationMaster} and the commands to be executed to run the
 * {@link ApplicationMaster}. <p>
 *
 * <p> Using the {@link ApplicationSubmissionContext}, the client submits the application to the
 * <code>ResourceManager</code> and then monitors the application by requesting the <code>ResourceManager</code>
 * for an {@link ApplicationReport} at regular time intervals. In case of the application taking too long, the client
 * kills the application by submitting a {@link KillApplicationRequest} to the <code>ResourceManager</code>. </p>
 */
public class Client {
    private static final Log LOG = LogFactory.getLog(Client.class);
    private static final Options OPTIONS = new Options();

    private static void addOption(String optName, boolean hasArg, String desc) {
        OPTIONS.addOption(null, optName, hasArg, desc);
    }

    static {
        addOption(OPT_NATIVE_LIBDIR, true, "Specify the directory for native libraries");
        addOption(OPT_APPNAME, true, "Application Name. Default value - DistributedShell");
        addOption(OPT_PRIORITY, true, "Application Priority. Default 0");
        addOption(OPT_QUEUE, true, "RM Queue in which this application is to be submitted");
        addOption(OPT_TIMEOUT, true, "Application timeout in milliseconds");
        addOption(OPT_APP_MASTER_MEMORY, true, "Amount of memory in MB to be requested to run the application master");
        addOption(OPT_APP_MASTER_VCORES, true, "Amount of virtual cores to be requested to run the application master");
        addOption(OPT_JAR, true, "Jar file containing the application master");
        addOption(OPT_CONTAINER_ENV, true, "Environment for application master script. Specified as env_key=env_val pairs");
        addOption(OPT_CONTAINER_MEMORY, true, "Amount of memory in MB to be requested to run containers");
        addOption(OPT_CONTAINER_VCORES, true, "Amount of virtual cores to be requested to run containers");
        addOption(OPT_NUM_CONTAINERS, true, "Number of containers on which the application shall run");
        addOption(OPT_CONTAINER_MASTER_CLASS, true, "Class of the master");
        addOption(OPT_CONTAINER_SLAVE_CLASS, true, "Class of the slave");

        addOption(OPT_LOG_PROPERTIES, true, "log4j.properties file");
        addOption(OPT_ATTEMPT_FAILURES_VALIDITY_INTERVAL, true,
                "when attempt_failures_validity_interval in milliseconds is set to > 0," +
                        "the failure number will not take failures which happen out of " +
                        "the validityInterval into failure count. " +
                        "If failure count reaches to maxAppAttempts, " +
                        "the application will be failed.");
        addOption(OPT_DEBUG, false, "Dump out debug information");
        addOption(OPT_DOMAIN, true, "ID of the timeline domain where the "
                + "timeline entities will be put");
        addOption(OPT_VIEW_ACLS, true, "Users and groups that allowed to "
                + "view the timeline entities in the given domain");
        addOption(OPT_MODIFY_ACLS, true, "Users and groups that allowed to "
                + "modify the timeline entities in the given domain");
        addOption(OPT_CREATE, false, "Flag to indicate whether to create the "
                + "domain specified with -domain.");
        addOption(OPT_HELP, false, "Print usage");
        addOption(OPT_NODE_LABEL_EXPRESSION, true,
                "Node label expression to determine the nodes"
                        + " where all the containers of this application"
                        + " will be allocated, \"\" means containers"
                        + " can be allocated anywhere, if you don't specify the option,"
                        + " default node_label_expression of queue will be used.");
    }

    // Configuration
    private Configuration conf;
    private YarnClient yarnClient;
    // Application master specific info to register a new Application with RM/ASM
    private String optAppName = "";
    // App master priority
    private int optAmPriority = 0;
    // Queue for App master
    private String optAmQueue = "";
    // Amt. of memory resource to request for to run the App Master
    private int optAmMemory = 10;
    // Amt. of virtual core resource to request for to run the App Master
    private int optAmVCores = 1;
    // Application master jar file
    private String optAppMasterJar = "";
    // Main class to invoke application master
    private final String appMasterMainClass;

    private String optContainerMasterClass;
    private String optContainerSlaveClass;


    private Map<String, String> optContainerEnv = new HashMap<String, String>();
    // Shell Command Container priority

    // Amt of memory to request for container in which shell script will be executed
    private int optContainerMemory = 10;
    // Amt. of virtual cores to request for container in which shell script will be executed
    private int optContainerVirtualCores = 1;
    // No. of containers in which the shell script needs to be executed
    private int optNumContainers = 1;
    private String optNodeLabelExpression = null;

    // log4j.properties file
    // if available, add to local resources and set into classpath
    private String optLog4jPropFile = "";

    // Start time for client
    private final long clientStartTime = System.currentTimeMillis();
    // Timeout threshold for client. Kill app after time interval expires.
    private long optClientTimeout = 600000;

    // flag to indicate whether to keep containers across application attempts.
    private long optAttemptFailuresValidityInterval = -1;

    // Debug flag
    boolean optDebugFlag = false;

    // Timeline domain ID
    private String optDomainId = null;

    // Flag to indicate whether to create the domain of the given ID
    private boolean optToCreateDomain = false;

    // Timeline domain reader access control
    private String optViewACLs = null;

    // Timeline domain writer access control
    private String optModifyACLs = null;

    private String optNativeLibDirectory = null;

    private String applicationHdfsDirectory = null;

    /**
     * @param args Command line arguments
     */
    public static void main(String[] args) {
        boolean result = false;
        try {
            Client client = new Client();
            LOG.info("Initializing Client");
            try {
                boolean doRun = client.init(args);
                if (!doRun) {
                    System.exit(0);
                }
            } catch (IllegalArgumentException e) {
                System.err.println(e.getLocalizedMessage());
                client.printUsage();
                System.exit(-1);
            }
            result = client.run();
            if (client.applicationHdfsDirectory != null)
                System.out.println("Application HDFS directory: " + client.applicationHdfsDirectory);
            if (result) {
                LOG.info("Application completed successfully");
                System.exit(0);
            }
        } catch (Throwable t) {
            LOG.fatal("Error running Client", t);
            System.exit(1);
        }
        LOG.error("Application failed to complete successfully");
        System.exit(2);
    }

    public Client() {
        this(new YarnConfiguration());
    }

    public Client(Configuration conf) {
        this(ApplicationMaster.class.getName(), conf);
    }

    Client(String appMasterMainClass, Configuration conf) {
        this.conf = conf;
        this.appMasterMainClass = appMasterMainClass;
        yarnClient = YarnClient.createYarnClient();
        yarnClient.init(conf);
    }


    /**
     * Helper function to print out usage
     */
    private void printUsage() {
        new HelpFormatter().printHelp("Client", OPTIONS);
    }

    /**
     * Parse command line options
     *
     * @param args Parsed command line options
     * @return Whether the init was successful to run the client
     * @throws ParseException
     */
    public boolean init(String[] args) throws ParseException {

        CommandLine cliParser = new GnuParser().parse(OPTIONS, args);

        if (args.length == 0) {
            throw new IllegalArgumentException("No args specified for client to initialize");
        }

        if (cliParser.hasOption(OPT_LOG_PROPERTIES)) {
            String log4jPath = cliParser.getOptionValue("OPT_LOG_PROPERTIES");
            try {
                Utils.updateLog4jConfiguration(Client.class, log4jPath);
            } catch (Exception e) {
                LOG.warn("Can not set up custom log4j properties. " + e);
            }
        }

        if (cliParser.hasOption(OPT_HELP)) {
            printUsage();
            return false;
        }

        optDebugFlag = cliParser.hasOption(OPT_DEBUG);

        optAppName = cliParser.getOptionValue(OPT_APPNAME, DEFAULT_APPNAME);
        optAmPriority = Utils.parseIntegerOption(cliParser, OPT_PRIORITY, DEFAULT_PRIORITY);
        optAmQueue = cliParser.getOptionValue(OPT_QUEUE, DEFAULT_QUEUE_ID);
        optAmMemory = Utils.parseIntegerOption(cliParser, OPT_APP_MASTER_MEMORY, DEFAULT_MASTER_MEMORY);
        optAmVCores = Utils.parseIntegerOption(cliParser, OPT_APP_MASTER_VCORES, DEFAULT_MASTER_VCORES);
        optNativeLibDirectory = cliParser.getOptionValue(OPT_NATIVE_LIBDIR, null);

        if (optAmMemory < 0) {
            throw new IllegalArgumentException("Invalid memory specified for application master, exiting."
                    + " Specified memory=" + optAmMemory);
        }
        if (optAmVCores < 0) {
            throw new IllegalArgumentException("Invalid virtual cores specified for application master, exiting."
                    + " Specified virtual cores=" + optAmVCores);
        }

        optAppMasterJar = Utils.getMandatoryOption(cliParser, OPT_JAR);


        Utils.parseKeyValueOption(cliParser, OPT_CONTAINER_ENV, optContainerEnv);

        optContainerMasterClass = Utils.getMandatoryOption(cliParser, OPT_CONTAINER_MASTER_CLASS);
        optContainerSlaveClass = Utils.getMandatoryOption(cliParser, OPT_CONTAINER_SLAVE_CLASS);
        optContainerMemory = Utils.parseIntegerOption(cliParser, OPT_CONTAINER_MEMORY, DEFAULT_CONTAINER_MEMORY);
        optContainerVirtualCores = Utils.parseIntegerOption(cliParser, OPT_CONTAINER_VCORES, DEFAULT_CONTAINER_VCORES);
        optNumContainers = Utils.parseIntegerOption(cliParser, OPT_NUM_CONTAINERS, DEFAULT_NUM_CONTAINERS);


        if (optContainerMemory < 0 || optContainerVirtualCores < 0 || optNumContainers < 1) {
            throw new IllegalArgumentException("Invalid no. of containers or container memory/vcores specified, exiting."
                    + " Specified optContainerMemory=" + optContainerMemory + ", optContainerVirtualCores=" + optContainerVirtualCores
                    + ", numContainer=" + optNumContainers);
        }

        optNodeLabelExpression = cliParser.getOptionValue(OPT_NODE_LABEL_EXPRESSION, null);

        optClientTimeout = Utils.parseIntegerOption(cliParser, OPT_TIMEOUT, DEFAULT_TIMEOUT);

        optAttemptFailuresValidityInterval = Utils.parseLongOption(cliParser,
                OPT_ATTEMPT_FAILURES_VALIDITY_INTERVAL, DEFAULT_ATTEMPT_FAILURES_VALIDITY_INTERVAL);

        optLog4jPropFile = cliParser.getOptionValue(OPT_LOG_PROPERTIES, "");

        // Get timeline domain options
        if (cliParser.hasOption(OPT_DOMAIN)) {
            optDomainId = cliParser.getOptionValue(OPT_DOMAIN);
            optToCreateDomain = cliParser.hasOption(OPT_CREATE);
            if (cliParser.hasOption(OPT_VIEW_ACLS)) {
                optViewACLs = cliParser.getOptionValue(OPT_VIEW_ACLS);
            }
            if (cliParser.hasOption(OPT_MODIFY_ACLS)) {
                optModifyACLs = cliParser.getOptionValue(OPT_MODIFY_ACLS);
            }
        }

        return true;
    }

    /**
     * Main run function for the client
     *
     * @return true if application completed successfully
     * @throws IOException
     * @throws YarnException
     */
    public boolean run() throws IOException, YarnException {
        LOG.info("Running Client");
        yarnClient.start();

        logClusterInfos();

        if (optDomainId != null && optDomainId.length() > 0 && optToCreateDomain) {
            prepareTimelineDomain();
        }

        // Get a new application id
        YarnClientApplication app = yarnClient.createApplication();
        GetNewApplicationResponse appResponse = app.getNewApplicationResponse();
        // TODO get min/max resource capabilities from RM and change memory ask if needed
        // If we do not have min/max, we may not be able to correctly request
        // the required resources from the RM for the app master
        // Memory ask has to be a multiple of min and less than max.
        // Dump out information about cluster capability as seen by the resource manager
        int maxMem = appResponse.getMaximumResourceCapability().getMemory();
        LOG.info("Max mem capabililty of resources in this cluster " + maxMem);

        // A resource ask cannot exceed the max.
        if (optAmMemory > maxMem) {
            LOG.info("AM memory specified above max threshold of cluster. Using max value."
                    + ", specified=" + optAmMemory
                    + ", max=" + maxMem);
            optAmMemory = maxMem;
        }

        int maxVCores = appResponse.getMaximumResourceCapability().getVirtualCores();
        LOG.info("Max virtual cores capabililty of resources in this cluster " + maxVCores);

        if (optAmVCores > maxVCores) {
            LOG.info("AM virtual cores specified above max threshold of cluster. "
                    + "Using max value." + ", specified=" + optAmVCores
                    + ", max=" + maxVCores);
            optAmVCores = maxVCores;
        }

        // set the application name
        ApplicationSubmissionContext appContext = app.getApplicationSubmissionContext();
        appContext.setKeepContainersAcrossApplicationAttempts(false);
        appContext.setApplicationName(optAppName);

        if (optAttemptFailuresValidityInterval >= 0) {
            appContext.setAttemptFailuresValidityInterval(optAttemptFailuresValidityInterval);
        }

        if (null != optNodeLabelExpression) {
            appContext.setNodeLabelExpression(optNodeLabelExpression);
        }


        // Set up the container launch context for the application master
        ApplicationId appId = appContext.getApplicationId();
        applicationHdfsDirectory = Utils.getFsApplicationSuffix(optAppName, appId);

        ContainerLaunchContext amContainer = buildAMContainerContext(appId);
        appContext.setAMContainerSpec(amContainer);


        // Set up resource type requirements
        // For now, both memory and vcores are supported, so we set memory and
        // vcores requirements
        Resource capability = Resource.newInstance(optAmMemory, optAmVCores);
        appContext.setResource(capability);

        // Set the priority for the application master
        // TODO - what is the range for priority? how to decide?
        Priority pri = Priority.newInstance(optAmPriority);
        appContext.setPriority(pri);

        // Set the queue to which this application is to be submitted in the RM
        appContext.setQueue(optAmQueue);

        // Submit the application to the applications manager
        // SubmitApplicationResponse submitResp = applicationsManager.submitApplication(appRequest);
        // Ignore the response as either a valid response object is returned on success
        // or an exception thrown to denote some form of a failure
        LOG.info("Submitting application to ASM");

        yarnClient.submitApplication(appContext);

        // TODO
        // Try submitting the same request again
        // app submission failure?

        // Monitor the application

        return monitorApplication(appId);
    }

    private Map<String, LocalResource> buildAMLocalResources(ApplicationId appId, FileSystem fs) throws IOException {
        // set local resources for the application master
        // local files or archives as needed
        // In this scenario, the jar file for the application master is part of the local resources
        Map<String, LocalResource> localResources = new HashMap<String, LocalResource>();

        LOG.info("Copy App Master jar from local filesystem and add to local environment");
        // Copy the application master jar to the filesystem
        // Create a local resource to point to the destination jar path
        Utils.addFileToLocalResources(optAppName, fs, optAppMasterJar, LOCAL_RSRC_APPLICATION_JAR, appId, localResources);

        // Set the log4j properties if needed
        if (!optLog4jPropFile.isEmpty()) {
            Utils.addFileToLocalResources(optAppName, fs, optLog4jPropFile, LOCAL_RSRC_LOG4J_PROPERTIES, appId, localResources);
        }

        return localResources;
    }

    private Map<String, String> buildAMEnvironment() {
        // Set the env variables to be setup in the env where the application master will be run
        LOG.info("Set the environment for the application master");
        Map<String, String> env = new HashMap<>();

        // Add AppMaster.jar location to classpath
        // At some point we should not be required to add
        // the hadoop specific classpaths to the env.
        // It should be provided out of the box.
        // For now setting all required classpaths including
        // the classpath to "." for the application jar
        StringBuilder classPathEnv = new StringBuilder(Environment.CLASSPATH.$$());

        classPathEnv.append(ApplicationConstants.CLASS_PATH_SEPARATOR).append("./*");
        for (String c : conf.getStrings(YarnConfiguration.YARN_APPLICATION_CLASSPATH,
                YarnConfiguration.DEFAULT_YARN_CROSS_PLATFORM_APPLICATION_CLASSPATH)) {
            classPathEnv.append(ApplicationConstants.CLASS_PATH_SEPARATOR);
            classPathEnv.append(c.trim());
        }
        classPathEnv.append(ApplicationConstants.CLASS_PATH_SEPARATOR).append("./log4j.properties");

        // add the runtime classpath needed for tests to work
        if (conf.getBoolean(YarnConfiguration.IS_MINI_YARN_CLUSTER, false)) {
            classPathEnv.append(ApplicationConstants.CLASS_PATH_SEPARATOR);
            classPathEnv.append(System.getProperty("java.class.path"));
        }

        env.put("CLASSPATH", classPathEnv.toString());

        return env;
    }

    private List<String> buildAMCommands() {
        LOG.info("Setting up app master command");

        String option = "--%s %s";
        ArrayList<String> args = new ArrayList<>();
        args.add(String.format(option, OPT_APPNAME, optAppName));
        args.add(String.format(option, OPT_CONTAINER_MASTER_CLASS, optContainerMasterClass));
        args.add(String.format(option, OPT_CONTAINER_SLAVE_CLASS, optContainerSlaveClass));
        args.add(String.format(option, OPT_CONTAINER_MEMORY, optContainerMemory));
        args.add(String.format(option, OPT_CONTAINER_VCORES, optContainerVirtualCores));
        args.add(String.format(option, OPT_NUM_CONTAINERS, optNumContainers));
        args.add(String.format(option, OPT_PRIORITY, optAmPriority));

        if (optNativeLibDirectory != null) {
            args.add(String.format(option, OPT_NATIVE_LIBDIR, optNativeLibDirectory));
        }

        for (Map.Entry<String, String> entry : optContainerEnv.entrySet()) {
            args.add(String.format(option, OPT_CONTAINER_ENV, entry.getKey() + "=" + entry.getValue()));
        }

        if (optDebugFlag) {
            args.add("--debug");
        }

        String command = Utils.createJavaCommand(optAmMemory, appMasterMainClass, "AppMaster", args);

        LOG.info("Completed setting up app master command " + command);
        List<String> result = new ArrayList<>();
        result.add(command);

        return result;
    }

    // Setup security tokens
    private ByteBuffer buildAMAuthenticationTokens(FileSystem fs) throws IOException {
        if (!UserGroupInformation.isSecurityEnabled()) {
            return null;
        }
        // Note: Credentials class is marked as LimitedPrivate for HDFS and MapReduce
        Credentials credentials = new Credentials();
        String tokenRenewer = conf.get(YarnConfiguration.RM_PRINCIPAL);
        if (tokenRenewer == null || tokenRenewer.length() == 0) {
            throw new IOException("Can't get Master Kerberos principal for the RM to use as renewer");
        }

        // For now, only getting tokens for the default file-system.
        final Token<?> tokens[] = fs.addDelegationTokens(tokenRenewer, credentials);
        if (tokens != null) {
            for (Token<?> token : tokens) {
                LOG.info("Got dt for " + fs.getUri() + "; " + token);
            }
        }
        DataOutputBuffer dob = new DataOutputBuffer();
        credentials.writeTokenStorageToStream(dob);
        ByteBuffer fsTokens = ByteBuffer.wrap(dob.getData(), 0, dob.getLength());

        return fsTokens;
    }

    private ContainerLaunchContext buildAMContainerContext(ApplicationId appId) throws IOException {
        FileSystem fs = FileSystem.get(conf);
        Map<String, LocalResource> localResources = buildAMLocalResources(appId, fs);
        Map<String, String> env = buildAMEnvironment();
        List<String> commands = buildAMCommands();
        ByteBuffer tokens = buildAMAuthenticationTokens(fs);

        // Set up the container launch context for the application master
        return ContainerLaunchContext.newInstance(localResources, env, commands, null, tokens, null);
    }

    /**
     * Monitor the submitted application for completion.
     * Kill application if time expires.
     *
     * @param appId Application Id of application to be monitored
     * @return true if application completed successfully
     * @throws YarnException
     * @throws IOException
     */
    private boolean monitorApplication(ApplicationId appId)
            throws YarnException, IOException {

        while (true) {
            // Check app status every 1 second.
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                LOG.debug("Thread sleep in monitoring loop interrupted");
            }

            // Get application report for the appId we are interested in
            ApplicationReport report = yarnClient.getApplicationReport(appId);

            LOG.info("Got application report from ASM for"
                    + ", appId=" + appId.getId()
                    + ", clientToAMToken=" + report.getClientToAMToken()
                    + ", appDiagnostics=" + report.getDiagnostics()
                    + ", appMasterHost=" + report.getHost()
                    + ", appQueue=" + report.getQueue()
                    + ", appMasterRpcPort=" + report.getRpcPort()
                    + ", appStartTime=" + report.getStartTime()
                    + ", yarnAppState=" + report.getYarnApplicationState().toString()
                    + ", distributedFinalState=" + report.getFinalApplicationStatus().toString()
                    + ", appTrackingUrl=" + report.getTrackingUrl()
                    + ", appUser=" + report.getUser());

            YarnApplicationState state = report.getYarnApplicationState();
            FinalApplicationStatus dsStatus = report.getFinalApplicationStatus();
            if (YarnApplicationState.FINISHED == state) {
                if (FinalApplicationStatus.SUCCEEDED == dsStatus) {
                    LOG.info("Application has completed successfully. Breaking monitoring loop");
                    return true;
                } else {
                    LOG.info("Application did finished unsuccessfully."
                            + " YarnState=" + state.toString() + ", DSFinalStatus=" + dsStatus.toString()
                            + ". Breaking monitoring loop");
                    return false;
                }
            } else if (YarnApplicationState.KILLED == state
                    || YarnApplicationState.FAILED == state) {
                LOG.info("Application did not finish."
                        + " YarnState=" + state.toString() + ", DSFinalStatus=" + dsStatus.toString()
                        + ". Breaking monitoring loop");
                return false;
            }

            if (System.currentTimeMillis() > (clientStartTime + optClientTimeout)) {
                LOG.info("Reached client specified timeout for application. Killing application");
                forceKillApplication(appId);
                return false;
            }
        }
    }

    /**
     * Kill a submitted application by sending a call to the ASM
     *
     * @param appId Application Id to be killed.
     * @throws YarnException
     * @throws IOException
     */
    private void forceKillApplication(ApplicationId appId)
            throws YarnException, IOException {
        // TODO clarify whether multiple jobs with the same app id can be submitted and be running at
        // the same time.
        // If yes, can we kill a particular attempt only?

        // Response can be ignored as it is non-null on success or
        // throws an exception in case of failures
        yarnClient.killApplication(appId);
    }

    private void prepareTimelineDomain() {
        TimelineClient timelineClient = null;
        if (conf.getBoolean(YarnConfiguration.TIMELINE_SERVICE_ENABLED,
                YarnConfiguration.DEFAULT_TIMELINE_SERVICE_ENABLED)) {
            timelineClient = TimelineClient.createTimelineClient();
            timelineClient.init(conf);
            timelineClient.start();
        } else {
            LOG.warn("Cannot put the domain " + optDomainId + " because the timeline service is not enabled");
            return;
        }
        try {
            //TODO: we need to check and combine the existing timeline domain ACLs,
            //but let's do it once we have client java library to query domains.
            TimelineDomain domain = new TimelineDomain();
            domain.setId(optDomainId);
            domain.setReaders(optViewACLs != null && optViewACLs.length() > 0 ? optViewACLs : " ");
            domain.setWriters(optModifyACLs != null && optModifyACLs.length() > 0 ? optModifyACLs : " ");
            timelineClient.putDomain(domain);
            LOG.info("Put the timeline domain: " + TimelineUtils.dumpTimelineRecordtoJSON(domain));
        } catch (Exception e) {
            LOG.error("Error when putting the timeline domain", e);
        } finally {
            timelineClient.stop();
        }
    }

    private void logClusterInfos() throws IOException, YarnException {
        YarnClusterMetrics clusterMetrics = yarnClient.getYarnClusterMetrics();
        LOG.info("Got Cluster metric info from ASM"
                + ", numNodeManagers=" + clusterMetrics.getNumNodeManagers());

        List<NodeReport> clusterNodeReports = yarnClient.getNodeReports(
                NodeState.RUNNING);
        LOG.info("Got Cluster node info from ASM");
        for (NodeReport node : clusterNodeReports) {
            LOG.info("Got node report from ASM for"
                    + ", nodeId=" + node.getNodeId()
                    + ", nodeAddress" + node.getHttpAddress()
                    + ", nodeRackName" + node.getRackName()
                    + ", nodeNumContainers" + node.getNumContainers());
        }

        QueueInfo queueInfo = yarnClient.getQueueInfo(this.optAmQueue);
        LOG.info("Queue info"
                + ", queueName=" + queueInfo.getQueueName()
                + ", queueCurrentCapacity=" + queueInfo.getCurrentCapacity()
                + ", queueMaxCapacity=" + queueInfo.getMaximumCapacity()
                + ", queueApplicationCount=" + queueInfo.getApplications().size()
                + ", queueChildQueueCount=" + queueInfo.getChildQueues().size());

        List<QueueUserACLInfo> listAclInfo = yarnClient.getQueueAclsInfo();
        for (QueueUserACLInfo aclInfo : listAclInfo) {
            for (QueueACL userAcl : aclInfo.getUserAcls()) {
                LOG.info("User ACL Info for Queue"
                        + ", queueName=" + aclInfo.getQueueName()
                        + ", userAcl=" + userAcl.name());
            }
        }
    }
}
