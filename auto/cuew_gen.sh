#!/bin/sh

SCRIPT=`realpath -s $0`
DIR=`dirname $SCRIPT`

python ${DIR}/cuew_gen.py hdr > $DIR/../include/cuew.h
python ${DIR}/cuew_gen.py impl > $DIR/../src/cuew.c
