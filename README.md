# End2EndSAT
Predicting Satisfiability via End-to-End Learning

- Add https://github.com/jhartford/AutoEncSets to python path
- Download dataset from https://www.cs.ubc.ca/labs/algorithms/Projects/End2EndSAT/data.zip
- Move the untarred data directory into the code directory
- From code directory, run (e.g., for size 100 var instances): python train.py 100 -m nn_raw -ps -cn -v