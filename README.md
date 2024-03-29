# End2EndSAT
Predicting Satisfiability via End-to-End Learning

See here of paper: https://www.cs.ubc.ca/labs/algorithms/Projects/End2EndSAT/paper.pdf

## Instructions (Linux)

Download repository and change directories:\
`git clone https://github.com/ChrisCameron1/End2EndSAT.git && cd End2EndSAT`

Download data:\
`wget https://www.cs.ubc.ca/labs/algorithms/Projects/End2EndSAT/data.zip && unzip data.zip`

We recommend to use with CometML for monitoring experiments. Please create a free cometML account at https://www.comet.com/site/. Open the `mypaths.py` file and replace with you username and key.

``` 
COMET={'un': '[your_username]',
       'key': '[your_key]'} 
```

Each directory in `data/` corresponds to a different dataset. To run with the hyper paramters from the paper on the `100` variable dataset, first untar that dataset (will take a while) and create corresponding `logs` directory:\
`tar -xvzf data/100.tar.gz && mv 100 data/ && mkdir logs && mkdir logs/100`

and run:\
`python train.py 100 -m nn_raw -ps -le 8 -lff 2 -u 128 -v`

Can add the `-cn` parameter for use without CometML but we do no support proper experimental logging in this setting.
