export JULIA_LOAD_PATH=$JULIA_LOAD_PATH:$PWD

if [ `uname` == "Darwin" ]; then
    export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$PWD/lib/
else
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/lib/
fi
