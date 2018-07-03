# recsys-challenge-2018

Solution for the 2018 Spotify RecSys Challenge by the team **Definitive Turtles**

### How to run
* place the million playlist dataset json files into the `data_raw/million_playlist_dataset` and the challenge set into the `data_raw/challenge_set` directories
* run the python files 1-6 without parameters
* run the script `7_rest.py` with parameters 0-4 (e.g. `for i in $(seq 0 4); do python 7_rest.py $i; done`)
* run `python 8_merge.py`
* run `python 9_format_and_fix.py output/merged.csv output/submission.csv`

### Requirements
* Python 3.5 with standard scientific packages (pandas, numpy, scipy, etc.)
* Either lot of processor cores or a lot of time. The scripts 5-6-7 contain a variable named `threads`; set this as desired
* About 40gb free space

### Reference environment
We ran the models using the following python version and packages:
```
Python 3.5.2 (we used the conda environment)
pandas 0.22.0
numpy 1.14.0
matplotlib 2.0.2
scipy 1.0.0
```
