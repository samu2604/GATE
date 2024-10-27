#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/bracket-0/stage-0

if [ $# -eq 0 ]; then
    FILE_NAME="best_result.txt"
else
    FILE_NAME=$1
fi

best_result=0.0
best_conf=""
current_wd=""

for configuration in $(ls)
do
 cd $configuration
 if $(ls | grep -q 'result'); then
    if (( $(bc <<<"$(cat result) < $best_result") )); then
        best_result=$(cat result);
        best_conf=$(cat config);
        current_wd=$(pwd)
    fi  
 fi  
 cd ..
done

var="text to append";
destdir=$SCRIPT_DIR/$FILE_NAME
 
echo "The best result is $best_result" >> $destdir
echo "The corresponding configuration is:" >> $destdir
echo $best_conf >> $destdir 
echo "The corresponding configuration directory is: $current_wd" >> $destdir
echo "" >> $destdir