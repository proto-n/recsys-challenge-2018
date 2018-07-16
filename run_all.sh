python 1_mpd_tocsv.py
python 2_cs_tocsv.py
python 3_tasks.py
python 4_0t.py
python 5_1t.py
python 6_100f.py
for i in $(seq 0 4); do python 7_rest.py $i; done
python 8_merge.py
python 9_format_and_fix.py output/merged.csv output/submission.csv