#!/bin/bash
FILES="./Already_Read/*"
rm Anystyle.bib
touch Anystyle.bib
for f in $FILES
do
  echo "Extracting biblography from $f"
  # take action on each file. $f store current file name
  #  cat $f
  anystyle -f bib find --no-layout $f >> Anystyle.bib 
  wc Anystyle.bib
done
