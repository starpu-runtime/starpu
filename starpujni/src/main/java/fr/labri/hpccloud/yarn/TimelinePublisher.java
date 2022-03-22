package fr.labri.hpccloud.yarn;

import org.apache.commons.logging.Log;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.security.UserGroupInformation;
import org.apache.hadoop.yarn.api.records.Container;
import org.apache.hadoop.yarn.api.records.ContainerStatus;
import org.apache.hadoop.yarn.api.records.timeline.TimelineEntity;
import org.apache.hadoop.yarn.api.records.timeline.TimelineEvent;
import org.apache.hadoop.yarn.api.records.timeline.TimelinePutResponse;
import org.apache.hadoop.yarn.client.api.TimelineClient;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.apache.hadoop.yarn.exceptions.YarnException;

import java.io.IOException;
import java.lang.reflect.UndeclaredThrowableException;
import java.security.PrivilegedExceptionAction;

public final class TimelinePublisher {
    public static final Log LOG = ApplicationMaster.LOG;

    private TimelineClient timelineClient;
    private UserGroupInformation appSubmitterUgi;

    protected TimelinePublisher(UserGroupInformation ugi, final Configuration conf)
            throws YarnException, IOException, InterruptedException {
        appSubmitterUgi = ugi;
        appSubmitterUgi.doAs(new PrivilegedExceptionAction<Void>() {
            @Override
            public Void run() throws Exception {
                if (conf.getBoolean(YarnConfiguration.TIMELINE_SERVICE_ENABLED,
                        YarnConfiguration.DEFAULT_TIMELINE_SERVICE_ENABLED)) {
                    // Creating the Timeline Client
                    timelineClient = TimelineClient.createTimelineClient();
                    timelineClient.init(conf);
                    timelineClient.start();
                } else {
                    timelineClient = null;
                    LOG.warn("Timeline service is not enabled");
                }
                return null;
            }
        });
    }

    public enum DSEvent {
        DS_APP_ATTEMPT_START, DS_APP_ATTEMPT_END, DS_CONTAINER_START, DS_CONTAINER_END
    }

    public enum DSEntity {
        DS_APP_ATTEMPT, DS_CONTAINER
    }

    public static TimelinePublisher startTimelineClient(UserGroupInformation appSubmitterUgi, final Configuration conf)
            throws YarnException, IOException, InterruptedException {
        try {
            return new TimelinePublisher(appSubmitterUgi, conf);
        } catch (UndeclaredThrowableException e) {
            throw new YarnException(e.getCause());
        }
    }

    public void stop() {
        if (timelineClient != null)
            timelineClient.stop();
    }

    public void publishContainerStartEvent(Container container, String domainId) {
        if (timelineClient == null)
            return;
        final TimelineEntity entity = new TimelineEntity();
        entity.setEntityId(container.getId().toString());
        entity.setEntityType(DSEntity.DS_CONTAINER.toString());
        entity.setDomainId(domainId);
        entity.addPrimaryFilter("user", appSubmitterUgi.getShortUserName());
        TimelineEvent event = new TimelineEvent();
        event.setTimestamp(System.currentTimeMillis());
        event.setEventType(DSEvent.DS_CONTAINER_START.toString());
        event.addEventInfo("Node", container.getNodeId().toString());
        event.addEventInfo("Resources", container.getResource().toString());
        entity.addEvent(event);

        try {
            appSubmitterUgi.doAs(new PrivilegedExceptionAction<TimelinePutResponse>() {
                @Override
                public TimelinePutResponse run() throws Exception {
                    return timelineClient.putEntities(entity);
                }
            });
        } catch (Exception e) {
            LOG.error("Container start event could not be published for " + container.getId().toString(),
                    e instanceof UndeclaredThrowableException ? e.getCause() : e);
        }
    }

    public void publishContainerEndEvent(ContainerStatus container, String domainId) {
        if (timelineClient == null)
            return;

        final TimelineEntity entity = new TimelineEntity();
        entity.setEntityId(container.getContainerId().toString());
        entity.setEntityType(DSEntity.DS_CONTAINER.toString());
        entity.setDomainId(domainId);
        entity.addPrimaryFilter("user", appSubmitterUgi.getShortUserName());
        TimelineEvent event = new TimelineEvent();
        event.setTimestamp(System.currentTimeMillis());
        event.setEventType(DSEvent.DS_CONTAINER_END.toString());
        event.addEventInfo("State", container.getState().name());
        event.addEventInfo("Exit Status", container.getExitStatus());
        entity.addEvent(event);
        try {
            timelineClient.putEntities(entity);
        } catch (YarnException | IOException e) {
            LOG.error("Container end event could not be published for " + container.getContainerId().toString(), e);
        }
    }

    public void publishApplicationAttemptEvent(String appAttemptId, DSEvent appEvent, String domainId) {
        if (timelineClient == null)
            return;

        final TimelineEntity entity = new TimelineEntity();
        entity.setEntityId(appAttemptId);
        entity.setEntityType(DSEntity.DS_APP_ATTEMPT.toString());
        entity.setDomainId(domainId);
        entity.addPrimaryFilter("user", appSubmitterUgi.getShortUserName());
        TimelineEvent event = new TimelineEvent();
        event.setEventType(appEvent.toString());
        event.setTimestamp(System.currentTimeMillis());
        entity.addEvent(event);
        try {
            timelineClient.putEntities(entity);
        } catch (YarnException | IOException e) {
            LOG.error("App Attempt " + (appEvent.equals(DSEvent.DS_APP_ATTEMPT_START) ? "start" : "end")
                    + " event could not be published for " + appAttemptId, e);
        }
    }

}
