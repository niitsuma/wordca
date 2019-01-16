#!/bin/bash
set -e

CORPUS=$1
VOCAB_MIN_COUNT=$2
#VOCAB_MIN_COUNT=5
#VOCAB_MIN_COUNT=100

WINDOW_SIZE=$3

VECTOR_SIZE=$4

VERBOSE=0
#MEMORY=4.0
MEMORY=20.0

VOCAB_FILE=$CORPUS-$VOCAB_MIN_COUNT.vocab.txt
if [ -e $VOCAB_FILE ]; then
    echo "$VOCAB_FILE found"
else
    build/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
fi


COOCCURRENCE_FILE=$CORPUS-$VOCAB_MIN_COUNT-$WINDOW_SIZE.gloveco.bin
if [ -e $COOCCURRENCE_FILE ]; then
    echo "$COOCCURRENCE_FILE found"
else
    build/cooccur   -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
fi

python3 wordca.py  $CORPUS $VOCAB_MIN_COUNT $WINDOW_SIZE $VECTOR_SIZE glove

