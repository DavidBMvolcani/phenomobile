#!/bin/bash

eval "$(conda shell.bash hook)" # we’re using bash for this script
conda activate  imgProc2
jupyter notebook

