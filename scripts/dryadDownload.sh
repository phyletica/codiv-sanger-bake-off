#! /usr/bin/env bash

# Steps this script will do:
## 1. Given a list of dois, download from Dryad
DRYAD=$1
TEMP_DIR=$2

## 2. unzip downloaded Dryad repo

while read item 
	do 
		author=`echo "$item" | awk -v FS="\t" '{ print $1}'`
	 
		#echo $author
		mkdir -p $TEMP_DIR/$author
		link=`echo "$item" | awk -v FS="\t" '{print $3}'`
		#echo $link
		
		cd $TEMP_DIR/$author
        	wget $link

	done<$DRYAD
