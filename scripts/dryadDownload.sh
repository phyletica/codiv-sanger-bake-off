#! /usr/bin/env bash

# Steps this script will do:

## 1. Given a list of dois, download from Dryad

	# Supply list of DOIs associated with the empirical studies with data we will be analyzing
	DRYAD_LIST=$1
	
	# Supply full path to directory where data is to be deposited: PATH-TO-YOUR-REPO-COPY/codiv-sanger-bake-off/data/empirical-datasets	
	DATA_DIR=$2

## 2. Create directories for those datasets, download the Dryad repos associated with each project, and unzip downloaded files

while read item 
	do 
		AUTHOR=`echo "$item" | awk -v FS="\t" '{ print $1}'`
	 
<<<<<<< HEAD
		mkdir -p $TEMP_DIR/$author
		link=`echo "$item" | awk -v FS="\t" '{print $3}'`
		
		cd $TEMP_DIR/$author
        	wget -O archive $link
            unzip archive
=======
		# Name subdirectory after author key
		mkdir -p $TEMP_DIR/$AUTHOR

		# Assign the DOI link as a variable
		LINK=`echo "$item" | awk -v FS="\t" '{print $3}'`
		
		cd $DATA_DIR/$AUTHOR
        	wget $LINK
>>>>>>> be5a0b1fe5313c009ecb2be2de1eda38122500eb
		

	done<$DRYAD
