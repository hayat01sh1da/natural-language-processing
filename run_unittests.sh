#!/bin/sh

for directory in $(ls -d */)
do
  cd $directory
    python -m unittest discover ./test
done
