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

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.*;
import java.util.Map.Entry;
import java.util.function.Function;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.io.IOUtils;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.hadoop.yarn.api.ApplicationConstants;
import org.apache.hadoop.yarn.api.records.ApplicationId;
import org.apache.hadoop.yarn.api.records.LocalResource;
import org.apache.hadoop.yarn.api.records.LocalResourceType;
import org.apache.hadoop.yarn.api.records.LocalResourceVisibility;
import org.apache.hadoop.yarn.util.ConverterUtils;
import org.apache.log4j.LogManager;
import org.apache.log4j.PropertyConfigurator;

import org.apache.commons.cli.Options;

public class Utils {

    public static void printUsage(String cmdLineSyntax, Options options) {
        new HelpFormatter().printHelp(cmdLineSyntax, options);
    }

    public static int parseIntegerOption(CommandLine cliParser, String option, int defaultValue) {
        return parseOptionValue(cliParser, option, defaultValue, Integer::parseInt);
    }

    public static long parseLongOption(CommandLine cliParser, String option, long defaultValue) {
        return parseOptionValue(cliParser, option, defaultValue, Long::parseLong);
    }

    public static <T> T parseOptionValue(CommandLine cliParser, String option, T defaultValue, Function<String, T> convert) {
        if (cliParser.hasOption(option)) {
            return convert.apply(cliParser.getOptionValue(option));
        }
        return defaultValue;
    }

    public static void parseKeyValueOption(CommandLine cliParser, String option, Map<String, String> result) {
        if(cliParser.hasOption(option)) {
            Utils.parseKeyValueOption(cliParser.getOptionValues(option), result);
        }
    }

    public static String getMandatoryOption(CommandLine cliParser, String option, String errMsg) {
        String result = cliParser.getOptionValue(option);
        if(result == null)
            throw new IllegalArgumentException(String.format (errMsg, option));
        return result;
    }

    public static String getMandatoryOption(CommandLine cliParser, String option) {
        return getMandatoryOption(cliParser, option, "no option --"+option+ " has been specified");
    }


    public static void parseKeyValueOption(String[] options, Map<String, String> result) {
        for (String kv : options) {
            kv = kv.trim();
            int index = kv.indexOf('=');
            if (index == -1) {
                result.put(kv, "");
                continue;
            }
            String key = kv.substring(0, index);
            String val = "";
            if (index < (kv.length() - 1)) {
                val = kv.substring(index + 1);
            }
            result.put(key, val);
        }
    }

    public static void updateLog4jConfiguration(Class<?> targetClass,
                                                String log4jPath) throws Exception {
        Properties customProperties = new Properties();
        FileInputStream fs = null;
        InputStream is = null;
        try {
            fs = new FileInputStream(log4jPath);
            is = targetClass.getResourceAsStream("/log4j.properties");
            customProperties.load(fs);
            Properties originalProperties = new Properties();
            originalProperties.load(is);
            for (Entry<Object, Object> entry : customProperties.entrySet()) {
                originalProperties.setProperty(entry.getKey().toString(), entry
                        .getValue().toString());
            }
            LogManager.resetConfiguration();
            PropertyConfigurator.configure(originalProperties);
        } finally {
            IOUtils.closeQuietly(is);
            IOUtils.closeQuietly(fs);
        }
    }


    public static String createJavaCommand (int vmMemory, String className, String outputPrefix, Iterable<String> args) {
        Vector<String> vargs = new Vector<>(30);
        // Set java executable command
        vargs.add(ApplicationConstants.Environment.JAVA_HOME.$$() + "/bin/java");
        // Set Xmx based on am memory size
        vargs.add("-Xmx" + vmMemory + "m");
        // Set class name
        vargs.add(className);
        // Set params
        for(String a : args)
            vargs.add(a);

        vargs.add("1>" + ApplicationConstants.LOG_DIR_EXPANSION_VAR + "/"+outputPrefix+".stdout");
        vargs.add("2>" + ApplicationConstants.LOG_DIR_EXPANSION_VAR + "/"+outputPrefix+".stderr");

        // Get final command
        StringBuilder command = new StringBuilder();
        for (String str : vargs) {
            command.append(str).append(" ");
        }
        return command.toString();
    }

    public static String getFsApplicationSuffix(String appName, ApplicationId appId) {
        return  appName + "/" + appId + "/";
    }

    public static String getFsApplicationDirectory(String appName, FileSystem fs, ApplicationId appId) {
        return new Path(fs.getHomeDirectory(), getFsApplicationSuffix(appName, appId)).toString();
    }

    public static Path getPathInAppDir(FileSystem fs, String appName, ApplicationId appId, String dstPath) {
        String suffix = appName + "/" + appId + "/" + dstPath;
        return new Path(fs.getHomeDirectory(), suffix);
    }

    public static void addFileToLocalResources(String appName, FileSystem fs, String fileSrcPath, String fileDstPath,
                                               ApplicationId appId,
                                               Map<String, LocalResource> localResources) throws IOException {
        Path dst = getPathInAppDir(fs, appName, appId, fileDstPath);
        Path src = new Path(fileSrcPath);
        System.err.println("SRC FILE="+src);
        System.err.println("DST FILE="+dst);
        fs.copyFromLocalFile(new Path(fileSrcPath), dst);
        addPathToLocalResources(fs, dst, localResources);
    }

    public static void addContentToLocalResources(String appName, FileSystem fs, String content, String fileDstPath,
                                                  ApplicationId appId,
                                                  Map<String, LocalResource> localResources) throws IOException {
        Path dst = getPathInAppDir(fs, appName, appId, fileDstPath);
        FSDataOutputStream ostream = null;
        try {
            ostream = FileSystem.create(fs, dst, new FsPermission((short) 0710));
            ostream.writeUTF(content);
        } finally {
            IOUtils.closeQuietly(ostream);
        }
        addPathToLocalResources(fs, dst, localResources);
    }

    public static void addDirToLocalResources(FileSystem fs, String dir, Map<String, LocalResource> localResources)
            throws IOException {
        Path dirPath = new Path(fs.getHomeDirectory()+Path.SEPARATOR+dir);
        FileStatus status = fs.getFileStatus(dirPath);
        if (! fs.exists(dirPath) || ! status.isDirectory() ) {
            throw new IOException("'"+dirPath+"' no such directory");
        }
        System.err.println("add files in directory "+status.getPath()+ " to local resources");
        for(FileStatus s : fs.listStatus(dirPath)) {
           if(! s.isFile())
               continue;
           Path p = s.getPath();

           addPathToLocalResources(fs, p, localResources);
        }
    }

    public static void addPathToLocalResources(FileSystem fs, Path path, Map<String, LocalResource> localResources)
            throws IOException {
        if (! fs.exists(path)) {
            throw new IOException(path + " no such file or directory");
        }
        FileStatus scFileStatus = fs.getFileStatus(path);
        System.err.println("add path "+scFileStatus.getPath()+ " to local resources");

        LocalResource scRsrc = LocalResource.newInstance(ConverterUtils.getYarnUrlFromURI(path.toUri()),
                LocalResourceType.FILE,
                LocalResourceVisibility.APPLICATION,
                scFileStatus.getLen(), scFileStatus.getModificationTime());
        localResources.put(path.getName(), scRsrc);
    }

    public static Map<String, String> checkEnvironmentVariables(String... variables) {
        Map<String, String> envs = System.getenv();
        for (String v : variables) {
            if (!envs.containsKey(v)) {
                throw new RuntimeException(v + "is not set in the environment");
            }
        }
        return envs;
    }

    public static String getMandatoryEnv(String variable) throws IOException {
        String result = System.getenv(variable);
        if (result == null) {
            throw new IOException("missing environment variable '" + variable + "'");
        }
        return result;
    }
}
