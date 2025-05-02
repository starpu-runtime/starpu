<!---
 StarPU --- Runtime system for heterogeneous multicore architectures.

 Copyright (C) 2022-2022    University of Bordeaux, CNRS (LaBRI UMR 5800), Inria

 StarPU is free software; you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation; either version 2.1 of the License, or (at
 your option) any later version.

 StarPU is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

 See the GNU Lesser General Public License in COPYING.LGPL for more details.
-->

# begin: macos only
# mettre la ligne suivante dans ~/.mavenrc pour sélectionner un JDK installé par l'utilisateur plutôt que le JDK de base
export JAVA_HOME=/Library/Java/JavaVirtualMachines/[...]/Contents/Home
# end: macos only

$ mvn package
$ java -jar build/linux-amd64/starpujni-1.0-SNAPSHOT.jar Pi3
$ java -jar build/linux-amd64/starpujni-1.0-SNAPSHOT.jar TransitiveClosure


$ mvn package -Pfat-jar
$ gdb ./build/linux-amd64/native/src/test/native/jvm-launcher
puis
run TransitiveClosure
ca segfaulte, lancer
continue

# Tests

## StarPU JNI Bindings

In order to run examples, use the script generated in `build/`*platform*`/scripts/run-example.sh`.

If maven is invoked with the goal `test-compile` a small C program 
`jvm-launcher` that behaves like `run-examples.sh` is compiled. The 
binary is located in directory: `build/`*platform*`/native/src/test/native/jvm-launcher`
In order to work properly the CLASSPATH has to reference external jar files likes Hadoop ones.
The list of Hadoop jar files is available as a classpath with the command: `hadoop classpath`
If necessary, the additional option `--glob` can be used to expand wildcards characters and 
get the actual list of jar files.

## Master/Slave application on Yarn

Once the project has been package (using `mvn package`), the directory `build/`*platform*`/scripts` should contain several scripts that deploys a
master/slave application. The application is registered as **Yaua** within the resource manager. For each run of the application, Yarn assigns
an identifier *appid* like *application_1551998931117_1357*. The application produces several files in the directory HDFS_HOME/Yaua/*appid*/ where
HDFS_HOME refers to the home directory of the user in the HDFS filesystem.

### useful yarn commands
- `yarn application -kill ` *appid*

### useful hdfs commands

- `hdfs dfs -h`
- `hdfs dfs -ls Yaua/`*appid*`/`
- `hdfs dfs -cat Yaua/`*appid*`/*.out`
- `hdfs dfs -rm -r Yaua/`*appid*`/`

### run-simple-on-yarn.sh
### run-pingpong-on-yarn.sh
### run-basic-starpu-on-yarn.sh
### run-pi-starpu-on-yarn.sh

# Web UI for Hadoop

Depending on the deployed version of Hadoop Web UI listen on different ports.
If Hadoop 3.1.0 has been spawned as a standalone local cluster:
- DFS NameNode: http://localhost:9870/
- Yarn Resource Manager: http://localhost:8088

The LSD platform uses an older version of Hadoop 2.7:
- Documentation on LSD platform: https://www.labri.fr/perso/flalanne/LSDDocumentation/
- DFS NameNode: https://h301.lsd.labri.fr:50470/ ou https://h306.lsd.labri.fr:50470/
- Yarn Resource Manager: https://h301.lsd.labri.fr:8090/ ou https://h304.lsd.labri.fr:50470/

