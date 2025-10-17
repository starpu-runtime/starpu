#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2010-2025  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
#
# StarPU is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# StarPU is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.

import os
import tests
import re
import argparse

class Command(object):
    def __init__(self, command):
        self.command = command

class Builder:
    def __init__(self, iname, name, host):
        self.steps = []
        self.env = {}
        self.iname = iname
        self.name = name
        self.host = host
        self.prologue = None
    def addStep(self, elem):
        self.steps.append(elem)
    def addPrologue(self, prologue):
        self.prologue = prologue

def create_builder(profile, profile_iname, profile_name, profile_host):
    p = Builder(profile_iname, profile_name, profile_host)

    ## build the environment
    p.env.update({'STARPU_HOME' : "$HOME/builds/" + profile_name + "/home"})
    p.env.update({'STARPU_FXT_PREFIX' : "$HOME/builds/" + profile_name + "/fxt"})
    p.env.update({'starpu_build_dir' : "$HOME/builds/" + profile_name + "/build"})
    p.env.update({'starpu_src_dir' : "$PWD"})
    p.env.update({'STARPU_OPENCL_PROGRAM_DIR' : "$PWD"})
    global_env = True
    if 'global_env' in profile.keys():
        global_env = profile['global_env']
    if global_env:
        p.env.update(tests.config['env'])
    if 'env' in profile.keys():
        p.env.update(profile['env'])

    bench = 'bench' in profile and profile['bench']
    if not bench:
        p.env.update({'STARPU_MICROBENCHS_DISABLED' : '1'})

    # We only read hwloc output, and trigger
    # https://github.com/open-mpi/hwloc/issues/394
    # on Fortran+OpenMPI
    if profile['hwloc_input'] is not None:
        p.env.update({'HWLOC_LIBXML': '0'})
        p.env.update({'STARPU_HWLOC_INPUT' : profile['hwloc_input']})

    # interpret @ variables in environment variables
    for key in p.env.keys() :
        m = re.split("@", p.env[key])
        for i in range(1,len(m),2):
            if m[i] in profile.keys():
                m[i] = profile[m[i]]
        p.env.update({key : ("".join(m))})

    p.addStep(Command(["set", "-x"]))
    p.addStep(Command(["set", "-e"]))

    p.addStep(Command([""]))
    p.addStep(Command(["STARPU_BRANCH=$CI_COMMIT_BRANCH"]))
    p.addStep(Command(["if test -z \"$STARPU_BRANCH\" ; then STARPU_BRANCH=$(git branch --show-current) ; fi"]))
    p.addStep(Command(["starpu_src_dir=$PWD"]))
    p.addStep(Command(["git config --global --add safe.directory $PWD"]))
    p.addStep(Command(['STARPU_GITVERSION=$(git log -n 1 --pretty="%H")']))
    p.addStep(Command(["starpu_artifacts=$starpu_src_dir/artifacts/$STARPU_GITVERSION"]))
    p.addStep(Command(["mkdir", "-p", "$starpu_artifacts"]))

    if bench:
        p.addStep(Command([""]))
        p.addStep(Command(["# set variables to store benchmarks results"]))
        p.addStep(Command(["export", "STARPU_BENCH_DIR=$starpu_artifacts/benchmarks"]))
        p.addStep(Command(["export", "STARPU_BENCH_ID=\"$(echo $(date +%Y-%m-%d) $STARPU_GITVERSION)\""]))
        p.addStep(Command(["mkdir", "-p", "$STARPU_BENCH_DIR"]))

    p.addStep(Command([""]))
    p.addStep(Command(["rm", "-rf", "$starpu_build_dir"]))
    p.addStep(Command(["mkdir", "-p", "$starpu_build_dir"]))
    p.addStep(Command(["("]))
    p.addStep(Command(["\techo", '"xoldPWD=\${PWD}"']))
    p.addStep(Command(["\tenv|grep -v LS_COLORS | grep -v DISPLAY | grep -v SSH_TTY | grep '^[A-Z]'|grep -v BASH_FUNC | grep '=' | sed 's/=/=\"/'| sed 's/$/\"/' | sed 's/^/export /'"]))
    p.addStep(Command(["\techo", '"cd \$xoldPWD"']))
    p.addStep(Command([")", ">", "$starpu_build_dir/env.sh"]))

    ## build the commands
    prologue = None
    if 'prologue' in profile.keys():
        prologue = profile['prologue']
    elif 'branch_prologue' in tests.config.keys():
        prologue = tests.config['branch_prologue']
    if prologue is not None:
        p.addPrologue(Command(prologue))

    if profile['release'] or bench:
        p.addStep(Command([""]))
        p.addStep(Command(["echo", "$STARPU_GITVERSION", ">", "$starpu_artifacts/../latest_release"]))
        p.addStep(Command(["echo", "$STARPU_BRANCH", ">", "$starpu_artifacts/branch_name"]))
        p.addStep(Command(["echo", str(profile['deploy']), ">", "$starpu_artifacts/deploy"]))
        p.addStep(Command([""]))

    p.addStep(Command(["./autogen.sh"]))
    p.addStep(Command(["cd", "$starpu_build_dir"]))

    configure_options = ["--enable-quick-check", "--disable-build-doc"]
    if bench:
        configure_options = ["--disable-build-doc"]
    global_opts = True
    if 'global_opts' in profile.keys():
        global_opts = profile['global_opts']
    if global_opts:
        configure_options += tests.config['opts']
    if 'opts' in profile.keys():
        configure_options += profile['opts']
    if 'coverage' in profile.keys():
        configure_options += profile['coverage']

    scan = profile['scan']
    if scan:
        p.addStep(Command(["mkdir", "scan"]))

    if 'configure_command' in profile.keys():
        profile_configure_command = profile['configure_command']
    else:
        profile_configure_command = ["$starpu_src_dir/configure"]

    if scan:
        configureCommand = ["scan-build", "--use-c++=/usr/bin/clang++", "-o", "scan"] + profile_configure_command
    else:
        configureCommand = profile_configure_command

    p.addStep(Command(configureCommand + configure_options + ["| " + "tee", "$starpu_artifacts/fulllog.txt"]))

    if scan:
        makeCommand = ["scan-build", "-o", "scan", "make", "-j", "32"]
    else:
        makeCommand = ["make", "-j", "32"]
    p.addStep(Command(makeCommand + ["| " + "tee", "-a", "$starpu_artifacts/fulllog.txt"]))

    if profile['release']:
        p.addStep(Command([""]))
        p.addStep(Command(["# copy release files in artifacts"]))
        p.addStep(Command(["make"] + ["dist"]))
        p.addStep(Command(["if test -f doc/README.org ; then mkdir -p $starpu_artifacts/doc ; cp doc/README.org $starpu_artifacts/doc ; fi"]))
        p.addStep(Command(["for doc in '' _dev _web_basics _web_extensions _web_faq _web_installation _web_introduction _web_languages _web_performances _web_applications ; do if test -f doc/doxygen${doc}/starpu${doc}.pdf ; then cp -p doc/doxygen${doc}/starpu${doc}.pdf $starpu_artifacts/doc ; fi ; if test -d doc/doxygen${doc}/html${doc} ; then cp -rp doc/doxygen${doc}/html${doc} $starpu_artifacts/doc ; fi ; done"]))
        p.addStep(Command(["TARBALLNAME=$(ls -a *.tar.gz|tail -1)"]))
        p.addStep(Command(["BASENAME=$(basename $TARBALLNAME .tar.gz)"]))
        p.addStep(Command(["DATE=$(date +%Y-%m-%d)"]))
        p.addStep(Command(["SUFFIX=\"_${DATE}-r$STARPU_GITVERSION.tar.gz\""]))
        p.addStep(Command(["NEWNAME=\"$BASENAME\"\"$SUFFIX\""]))
        p.addStep(Command(["cp $TARBALLNAME $starpu_artifacts/$NEWNAME"]))
        p.addStep(Command([""]))

    if bench:
        p.addStep(Command([""]))
        p.addStep(Command(["mkdir", "-p", "$starpu_artifacts/benchmarks"]))
        p.addStep(Command(["# make sure perf models are set from zero"]))
        p.addStep(Command(["rm", "-rf", "$STARPU_HOME"]))
        p.addStep(Command([""]))

    p.addStep(Command(["touch", "$starpu_artifacts/make_showsuite.txt"]))
    p.addStep(Command(["touch", "$starpu_artifacts/make_showcheck.txt"]))
    p.addStep(Command(["ret=0"]))
    p.addStep(Command([""]))

    if profile['deploy']:
        parallel = []
        if 'parallel' in profile.keys():
            parallel = ["-j", str(profile['parallel'])]
        restrict = []
        if 'rcheck' in profile.keys():
            restrict = profile['rcheck']
        checkCommand = ["make"] + parallel + ["--keep-going", "check"] + restrict

        p.addStep(Command(["# is make check disabled?"]))
        p.addStep(Command(["if test \"$1\" == \"-x\" ; then exit $ret ; fi"]))
        p.addStep(Command([""]))

        p.addStep(Command(["set +e"]))
        p.addStep(Command(["(", "set", "-o", "pipefail", ";"] + checkCommand + ["| " + "tee", "-a", "$starpu_artifacts/fulllog.txt", ")"]))
        p.addStep(Command(["ret=$?"]))
        p.addStep(Command([""]))

        if profile['showsuite']:
            p.addStep(Command(["make", "showsuite"] + restrict + [">", "$starpu_artifacts/make_showsuite.txt"]))
        p.addStep(Command(["make", "showcheck"] + restrict + [">", "$starpu_artifacts/make_showcheck.txt"]))

        p.addStep(Command(["cat", "$starpu_artifacts/make_showsuite.txt", ">>", "$starpu_artifacts/fulllog.txt"]))
        p.addStep(Command(["cat", "$starpu_artifacts/make_showcheck.txt", ">>", "$starpu_artifacts/fulllog.txt"]))

    if profile['deploy']:
        p.addStep(Command([""]))
        p.addStep(Command(["make"] + ["showfailed"] + restrict + ["| " + "tee", "-a", "$starpu_artifacts/fulllog.txt"]))

    if 'coverage' in profile.keys():
        p.addStep(Command([""]))
        p.addStep(Command(["lcov", "--directory", ".", "--capture", "--output", "coverage.info"]))
        p.addStep(Command(["genhtml", "--output-directory", "coverage", "coverage.info"]))
        if profile['release']:
            p.addStep(Command(["cp", "coverage.info", "$starpu_artifacts/starpu_${STARPU_GITVERSION}.lcov"]))
            p.addStep(Command(["cp", "-rp", "coverage", "$starpu_artifacts/coverage_$STARPU_GITVERSION"]))
        p.addStep(Command([""]))

    p.addStep(Command(["for x  in $(find . -name \"*log\") ; do mkdir -p logs/$(dirname $x) && cp $x logs/$(dirname $x) ; done > /dev/null 2>&1"]))
    p.addStep(Command(["make", "clean", ">", "/dev/null", "2>&1"]))
    if profile['ignore_fail']:
        p.addStep(Command(["exit", "0"]))
    else:
        p.addStep(Command(["exit", "$ret"]))
    return p

def get_builder_name(host, scan, profile_name):
    return host + '_' + scan + profile_name

def builderset():
    builderset = []
    for profile in tests.profiles:
        if 'showsuite' not in profile.keys():
            profile['showsuite'] = True

        if 'deploy' not in profile.keys():
            profile['deploy'] = False

        if 'release' not in profile.keys():
            profile['release'] = False

        if 'ignore_fail' not in profile.keys():
            profile['ignore_fail'] = False

        if 'scan' not in profile.keys():
            profile['scan'] = False

        scan_name = ''
        if profile['scan']:
            scan_name = 'scan-'

        if 'hwloc_input' not in profile.keys():
            profile['hwloc_input'] = "/mnt/scratch/benchmarks/buildbot/hwloc.xml"
        else:
            if type(profile['hwloc_input']) is not type("x"):
                profile['hwloc_input'] = None

        for host in profile['hosts']:
            builderset.append(create_builder(profile,
                                             profile['name'],
                                             get_builder_name(host, scan_name, profile['name']),
                                             host)
                              )

    return builderset

##########################
#
##########################

parser = argparse.ArgumentParser()
parser.add_argument('-hh', '--host')
parser.add_argument('-p', '--profile')
args = parser.parse_args()

for builder in builderset():
    if args.host is not None:
        if (args.host != builder.host):
            continue

    if args.profile is not None:
        if (args.profile != builder.iname):
            continue

    dirname = "./ci/" + builder.host
    os.makedirs(dirname, exist_ok=True)
    script_name =  dirname + "/" + builder.name + ".sh"
    fout = open(script_name, "w")
    fout.write("#!/bin/bash\n\n")
    fout.write("echo \"#PROFILE:" + builder.host + ":" + builder.name + "\"\n")

    fout.write("\n")
    if builder.prologue is not None:
        prologue = " ".join(builder.prologue.command)
        fout.write(prologue + "\n")

    fout.write("\n")
    for key in builder.env:
        fout.write("export " + key + "=\"" + builder.env[key] + "\"\n")
    fout.write("\n")

    for step in builder.steps:
        if type(step.command) is type(""):
            fout.write(step.command + "\n")
        else:
            command = " ".join(step.command)
            fout.write(command + "\n")

    fout.close()
    if args.host is not None:
        print(builder.iname)
    elif args.profile is not None:
        print(script_name)
    else:
        print("Profile <%s> --> script <%s>" % (builder.name, script_name))
