#!/bin/bash
for SETUP in beta delta sigma 
do
    python3 joiner.py --file mses/ERM-${SETUP}.txt --dir ERM-${SETUP}/
done