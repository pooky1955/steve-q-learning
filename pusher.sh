#!/bin/bash
while true
do 
  echo "changed"
  git add . && git commit -m "update" && git push origin liveshare
  sleep 10
done
