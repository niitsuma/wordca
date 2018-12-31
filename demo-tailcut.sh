#!/bin/bash
set -e

make
if [ ! -e text8 ]; then
  if hash wget 2>/dev/null; then
    wget http://mattmahoney.net/dc/text8.zip
  else
    curl -O http://mattmahoney.net/dc/text8.zip
  fi
  unzip text8.zip
  rm text8.zip
fi

make

BUILDDIR=build
for CORPUS in text01 text8 ; do
    echo $CORPUS

    VOCAB_MIN_COUNT=5
    WINDOW_SIZE_MAX=24
    VECTOR_SIZE=8000
    bash tailcut.sh  $CORPUS  $VOCAB_MIN_COUNT $WINDOW_SIZE_MAX $VECTOR_SIZE
    #python3 wordca.py  $CORPUS $VOCAB_MIN_COUNT $WINDOW_SIZE_MAX 1000 tailcut 

done 
