
package fr.labri.hpccloud.yarn;

/**
 * Constants used in both Client and Application Master
 */
public class Constants {
  public static String OPT_NATIVE_LIBDIR = "native-libdir";

  public static String OPT_APPNAME = "appname";
  public static String OPT_PRIORITY = "priority";
  public static String OPT_QUEUE = "queue";
  public static String OPT_TIMEOUT = "timeout";

  public static String OPT_APP_MASTER_MEMORY = "master_memory";
  public static String OPT_APP_MASTER_VCORES = "master_vcores";

  public static String OPT_JAR = "jar";

  public static String OPT_CONTAINER_ENV = "container-env";
  public static String OPT_CONTAINER_MEMORY = "container-memory";
  public static String OPT_CONTAINER_VCORES = "container-vcores";
  public static String OPT_NUM_CONTAINERS = "num-containers";
  public static String OPT_CONTAINER_MASTER_CLASS = "container-master-class";
  public static String OPT_CONTAINER_SLAVE_CLASS = "container-slave-class";


  public static String OPT_LOG_PROPERTIES = "log_properties";
  public static String OPT_ATTEMPT_FAILURES_VALIDITY_INTERVAL = "attempt_failures_validity_interval";
  public static String OPT_DEBUG = "debug";
  public static String OPT_DOMAIN = "domain";
  public static String OPT_VIEW_ACLS = "view_acls";
  public static String OPT_MODIFY_ACLS = "modify_acls";
  public static String OPT_CREATE = "create";
  public static String OPT_HELP = "help";
  public static String OPT_NODE_LABEL_EXPRESSION = "node_label_expression";
  public static String OPT_APP_ATTEMPT_ID = "app_attempt_id";


  public static String LOCAL_RSRC_LOG4J_PROPERTIES = "log4j.properties";
  public static String LOCAL_RSRC_APPLICATION_JAR = "starpu-masterslave.jar";

  public static int DEFAULT_MASTER_MEMORY = 512;
  public static int DEFAULT_MASTER_VCORES = 1;
  public static int DEFAULT_CONTAINER_MEMORY = 10;
  public static int DEFAULT_CONTAINER_VCORES = 1;
  public static int DEFAULT_NUM_CONTAINERS = 2;
  public static int DEFAULT_PRIORITY = 0;
  public static int SECOND = 1000;
  public static int MINUTE = 60 * SECOND;
  public static int DEFAULT_TIMEOUT = 10*MINUTE;
  public static String DEFAULT_QUEUE_ID = "default";

  public static long DEFAULT_ATTEMPT_FAILURES_VALIDITY_INTERVAL = -1;

  public static String DEFAULT_APPNAME = "Yaua";

  public static String ENV_APPLICATION_SLAVE_ID = "APPLICATION_SLAVE_ID";
  public static String ENV_APPLICATION_MASTER_HOSTNAME = "APPLICATION_MASTER_HOSTNAME";
  public static String ENV_APPLICATION_DIRECTORY = "APPLICATION_DIRECTORY";
}
