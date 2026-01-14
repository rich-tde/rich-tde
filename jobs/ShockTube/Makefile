# Global variables
PL = 1
DL = 1
PR = 0.1
DR = 0.125

RICH_PROBLEM = ShockTube
LMOD_ENV = new_rich_build

RICH_PATH = /home/hey4/RICH-fwrk
DATA_PATH = /home/hey4/data/raw
OUTPUT_PATH = $(DATA_PATH)/PL$(PL)PR$(PR)DL$(DL)DR$(DR)

## note: $(@D) gives the path to the output file
## $_ uses the same argument as the last command
## && makes them run in one shell (make executes every line in a new shell)
$(OUTPUT_PATH)/snap_%.h5 : snap_%.h5
	mkdir -p $(@D) && cp snap_%.h5 $_ 

snap_final.h5: run.sh $(RICH_PATH)/build/rich leftpressure.txt leftdensity.txt rightpressure.txt rightdensity.txt
	./run.sh

$(RICH_PATH)/build/rich: $(RICH_PATH)/runs/ShockTube/main.cpp build.sh
	./build.sh

run.sh:
	printf '%s\n' \
		'#!/bin/bash -l' \
		'module purge' \
		'module restore $(LMOD_ENV)' \
		'$(RICH_PATH)/build/rich' \
	> $@
	chmod +x $@

build.sh:
	printf '%s\n' \
		'#!/bin/bash -l' \
		'rm -r $(RICH_PATH)/build' \
		'module purge' \
		'module restore $(LMOD_ENV)' \
		'$(RICH_PATH)/config.py --problem=$(RICH_PROBLEM)' \
		'$(MAKE) -C $(RICH_PATH)' \
	> $@
	chmod +x $@

leftpressure.txt:
	echo $(PL) > $@

leftdensity.txt:
	echo $(DL) > $@

rightpressure.txt:
	echo $(PR) > $@

rightdensity.txt:
	echo $(DR) > $@
