#!/bin/bash
set -e

CORPUS=$1
VOCAB_MIN_COUNT=$2
#VOCAB_MIN_COUNT=5
#VOCAB_MIN_COUNT=100

WINDOW_SIZE_MAX=$3
#WINDOW_SIZE_MAX=4

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

for ((WINDOW_SIZE=1; WINDOW_SIZE <= $WINDOW_SIZE_MAX; WINDOW_SIZE++)); do 
    #WINDOW_SIZE=1
    COOCCURRENCE_FILE=$CORPUS-$VOCAB_MIN_COUNT-$WINDOW_SIZE.cooccurrence.bin
    if [ -e $COOCCURRENCE_FILE ]; then
	echo "$COOCCURRENCE_FILE found"
    else
	build/wordcooccurleftfixlen  -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
    fi
done

python3 wordca.py  $CORPUS $VOCAB_MIN_COUNT $WINDOW_SIZE_MAX $VECTOR_SIZE tailcut 
