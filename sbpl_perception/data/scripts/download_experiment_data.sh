#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd $DIR

HEURISTICS_FILE=heuristics.tar.gz
EXPERIMENTS_FILE=experiment_input.tar.gz
HEURISTICS_FILE_CHECKSUM=b97722cd4a20bd6cb8fc34e2c8aeff67
EXPERIMENTS_FILE_CHECKSUM=0e9840a8d9dd2d3f39e9469d0d625b75

i=0
orig_checksum=$EXPERIMENTS_FILE_CHECKSUM
for FILE in $EXPERIMENTS_FILE $HEURISTICS_FILE; do

	URL=https://sbpl.net/shared/Venkat/sbpl_perception/data/$FILE

	if [ "$i" != "0" ]; then
    # Place the heuristics folder under sbpl_perception
    cd ..
		orig_checksum=$HEURISTICS_FILE_CHECKSUM
	fi
	let i=i+1

	if [ -f $FILE ]; then
		echo "File already exists. Checking md5..."
		os=`uname -s`
		if [ "$os" = "Linux" ]; then
			checksum=`md5sum $FILE | awk '{ print $1 }'`
		elif [ "$os" = "Darwin" ]; then
			checksum=`cat $FILE | md5`
		fi
		if [ "$checksum" = "$orig_checksum" ]; then
			echo "Checksum is correct. No need to download."
			continue
		else
			echo "Checksum is incorrect. Need to download again."
		fi
	fi

	echo "Downloading file $FILE"
	wget $URL -O $FILE --no-check-certificate
	echo "Unzipping..."

	tar zxvf $FILE
done
