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

CORPUS=text01   ###small debug data
#CORPUS=text8

VOCAB_FILE=vocab.txt
COOCCURRENCE_FILE=cooccurrence.bin
COOCCURRENCE_SHUF_FILE=cooccurrence.shuf.bin
BUILDDIR=build


VOCAB_MIN_COUNT=5
WINDOW_SIZE=30

VERBOSE=0
MEMORY=20.0
VOCAB_MIN_COUNT=5

WINDOW_SIZE_MAX=30



VOCAB_FILE=$CORPUS-$VOCAB_MIN_COUNT.vocab.txt
$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE

COOCCURRENCE_FILE=$CORPUS-$VOCAB_MIN_COUNT-$WINDOW_SIZE.cooccurrence.bin

$BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE


python3 wordca.py

