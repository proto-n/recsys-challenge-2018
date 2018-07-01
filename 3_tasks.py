import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from math import log

pdata = pd.read_csv('data/challenge_set/playlists.csv')
data = pd.read_csv('data/challenge_set/playlist_meta.csv')

data['task'] = ""
data.loc[data.name.isnull() & (data.num_samples == 5), "task"] = "5nt"
data.loc[~data.name.isnull() & (data.num_samples == 5), "task"] = "5t"
data.loc[data.name.isnull() & (data.num_samples == 10), "task"] = "10nt"
data.loc[~data.name.isnull() & (data.num_samples == 10), "task"] = "10t"
data.loc[data.num_samples == 0, "task"] = "0t"
data.loc[data.num_samples == 1, "task"] = "1t"

for i in [25, 100]:
    pdata_with_num = pdata[pdata.pid.isin(data[data.num_samples == i].pid)]
    rand_ids = pdata_with_num[pdata_with_num.pos >= i].pid.unique()
    data.loc[data.num_samples == i, "task"] = str(i) + "f"
    data.loc[data.pid.isin(rand_ids), "task"] = str(i) + "r"

data.to_csv('data/challenge_set/playlist_meta_tasks.csv', index=False)
