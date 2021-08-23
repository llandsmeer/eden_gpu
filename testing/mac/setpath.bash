#!/bin/bash

# Create a temporary folder, relative to cwd, to store links to the binaries to be used
# something more reasonable LATER

TEMP_LINK_DIR=$(pwd)/build_aliases

mkdir -p $TEMP_LINK_DIR

# These are based on Homebrew with a specific version, for now
# TODO select version of gcc
HOMEBREW_PATH=$(brew --prefix)
HOMEBREW_BIN="$HOMEBREW_PATH/bin"
ln -fs "$HOMEBREW_BIN/gcc-11" "$TEMP_LINK_DIR/cc"
ln -fs "$HOMEBREW_BIN/gcc-11" "$TEMP_LINK_DIR/gcc"
ln -fs "$HOMEBREW_BIN/g++-11" "$TEMP_LINK_DIR/g++"
ln -fs "$HOMEBREW_BIN/c++-11" "$TEMP_LINK_DIR/c++"
ln -fs "$HOMEBREW_BIN/cpp-11" "$TEMP_LINK_DIR/cpp"

ln -fs "$HOMEBREW_PATH/opt/bison/bin/bison" "$TEMP_LINK_DIR/bison"
ln -fs "$HOMEBREW_PATH/opt/flex/bin/flex"   "$TEMP_LINK_DIR/flex"

PATH="$TEMP_LINK_DIR:$PATH"
