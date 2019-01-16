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



bash gloveca.sh  text01 5  30  100
bash gloveca.sh  text8  5  30  100

bash tailcut.sh  text01 5  24  1000
bash tailcut.sh  text8  5  24  8000


