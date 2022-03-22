package fr.labri.hpccloud.yarn;

import org.apache.commons.logging.Log;
import org.apache.hadoop.yarn.api.ContainerManagementProtocol;
import org.apache.hadoop.yarn.api.records.*;

/**
 * Thread to connect to the {@link ContainerManagementProtocol} and launch the container
 * that will execute the shell command.
 */
public class LaunchContainerThread extends Thread {
    public static final Log LOG = ApplicationMaster.LOG;

    // Allocated container
    private Container container;
    private ApplicationMaster am;
    private ContainerLaunchContext context;

    /**
     * @param lcontainer        Allocated container
     * @param containerListener Callback handler of the container
     */
    public LaunchContainerThread(Container lcontainer, ApplicationMaster containerListener,
                                 ContainerLaunchContext ctx) {
        this.container = lcontainer;
        this.am = containerListener;
        this.context = ctx;
    }

    @Override
    /**
     * Connects to CM, sets up container launch context
     * for shell command and eventually dispatches the container
     * start request to the CM.
     */
    public void run() {
        LOG.info("Setting up container launch container for containerid=" + container.getId());

        am.addContainer(container.getId(), container);
        am.getNmClient().startContainerAsync(container, context);
    }
}
