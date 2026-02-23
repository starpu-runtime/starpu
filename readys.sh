export LIBTORCH_INSTALL_PATH=/usr/local
export READYS_DEPENDENCIES=$HOME/readys_dependencies
export LIBTORCH_EXTENSIONS_INSTALL_PATH=/usr/local
mkdir -p $READYS_DEPENDENCIES/sources
cd $READYS_DEPENDENCIES/sources
git clone --recurse-submodules https://github.com/rusty1s/pytorch_scatter.git
mkdir -p $READYS_DEPENDENCIES/builds/pytorch_scatter_build
cd $READYS_DEPENDENCIES/builds/pytorch_scatter_build
sudo apt-get install cmake
sudo apt-get install libtorch-dev
cmake -DCMAKE_PREFIX_PATH="$(python3-config --prefix)" -DCMAKE_INSTALL_PREFIX:PATH=$LIBTORCH_EXTENSIONS_INSTALL_PATH $READYS_DEPENDENCIES/sources/pytorch_scatter
