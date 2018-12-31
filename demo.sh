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
#for CORPUS in text01  text8 ; do
for CORPUS in text01 ; do

    echo $CORPUS

    VOCAB_MIN_COUNT=5
    WINDOW_SIZE=30

    VERBOSE=0
    MEMORY=20.0

    VOCAB_FILE=$CORPUS-$VOCAB_MIN_COUNT.vocab.txt
    #$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
    if [ -e $VOCAB_FILE ]; then
	echo "$VOCAB_FILE found"
    else
	$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
    fi

    COOCCURRENCE_FILE=$CORPUS-$VOCAB_MIN_COUNT-$WINDOW_SIZE.gloveco.bin
    if [ -e $COOCCURRENCE_FILE ]; then
	echo "$COOCCURRENCE_FILE found"
    else
	$BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
    fi
    
    python3 wordca.py  $CORPUS $VOCAB_MIN_COUNT $WINDOW_SIZE 1000 glove

done 
