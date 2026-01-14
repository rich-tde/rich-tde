#!/bin/bash
#Quick helper script that compiles and runs test
#Used for quickly testing out different configs
module purge
module restore new_rich_build

cd /home/hey4/RICH-fwrk

rm -r build	# remove previous build

./config.py --problem=SedovDissipation

make -j

cd build

./rich
