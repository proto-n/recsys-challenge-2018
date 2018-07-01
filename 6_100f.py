import pandas as pd
import numpy as np
import scipy.sparse as spl
from concurrent.futures import ProcessPoolExecutor
import sys

threads = 20
all_tasks = [
    [100, 40000, ['100f'], 0.44, 2, 30],
]

split, knn_k, test_task, powb, dew, lowbar = all_tasks[0]


def recode(column, min_val=0):
    uniques = column.unique()
    codes = range(min_val, len(uniques) + min_val)
    code_map = dict(zip(uniques, codes))
    return (column.map(code_map), code_map)


def reverse_code(column, code_map):
    inv_map = {v: k for k, v in code_map.items()}
    return column.map(inv_map)


playlist_meta = pd.read_csv('data/million_playlist_dataset/playlist_meta.csv')
playlist_meta_c = pd.read_csv('data/challenge_set/playlist_meta.csv')
playlist_meta = pd.concat([playlist_meta, playlist_meta_c], axis=0, ignore_index=True)
song_meta = pd.read_csv('data/million_playlist_dataset/song_meta_no_duplicates.csv')
playlist_meta['pid_code'], pid_codes = recode(playlist_meta['pid'])
song_meta['song_code'], song_codes = recode(song_meta['song_id'])

train = pd.read_csv('data/million_playlist_dataset/playlists.csv')
test = pd.read_csv('data/challenge_set/playlists.csv')

test_tasks = pd.read_csv('data/challenge_set/playlist_meta_tasks.csv')
test_tasks_pids = test_tasks[test_tasks.task.isin(test_task)].pid.unique()

test = test[test.pid.isin(test_tasks_pids)].copy()


train['pid_code'] = train['pid'].map(pid_codes)
train['song_code'] = train['song_id'].map(song_codes)
train.sort_values(['pid_code', 'song_code'], inplace=True)
test['pid_code'] = test['pid'].map(pid_codes)
test['song_code'] = test['song_id'].map(song_codes)

train_agg = train.drop_duplicates(subset=['pid_code', 'song_code']).copy()
test_agg = test.drop_duplicates(subset=['pid_code', 'song_code']).copy()
train_agg['val'] = 1
test_agg['val'] = 1

train_agg['val_stoch'] = train_agg.groupby('pid_code').val.transform(lambda x: x / np.linalg.norm(x))
test_agg['val_stoch'] = test_agg.groupby('pid_code').val.transform(lambda x: x / np.linalg.norm(x))

test_agg_pop = test_agg.join(train.song_code.value_counts().rename('pop'), on='song_code')
test_agg_pop['pop'].fillna(1, inplace=True)


sp_A = spl.coo_matrix((train_agg['val_stoch'].values.T, train_agg[['pid_code', 'song_code']].values.T))
sp_A._shape = (int(playlist_meta.pid_code.max() + 1), int(song_meta.song_code.max() + 1))
sp_A = sp_A.tocsr()
sp_A_t = sp_A.T
sp_A_const = spl.coo_matrix((train_agg['val'].values.T, train_agg[['pid_code', 'song_code']].values.T))
sp_A_const._shape = (int(playlist_meta.pid_code.max() + 1), int(song_meta.song_code.max() + 1))
sp_A_const = sp_A_const.tocsr()
sp_A_const_t = sp_A_const.T


plusadd = 0

def recs_for_ids(ids_):
    dfs = []
    ndcgs = []
    for pid_ in ids_:
        p1 = test_agg_pop[(test_agg_pop.pid_code == pid_)]
        p1_train = p1[p1['pos'] >= 0].copy()

        poses = p1_train['pos']
        poses = np.maximum(poses, lowbar)
        p1_train['val_stoch'] = p1_train['val_stoch'] * (1 + poses * (1 / dew))

        np_p1 = np.zeros([int(song_meta.song_code.max() + 1), 1])
        np_p1[p1.song_code.values] = p1_train[['val_stoch']].values / ((p1_train[['pop']].values - 1)**(powb) + 1)
        simpls = sp_A.dot(np_p1)
        simpls2 = np.zeros_like(simpls)
        inds = simpls.reshape(-1).argsort()[-knn_k:][::-1]
        vals = simpls[inds]
        m = np.max(vals)
        if(m == 0):
            m += 0.01
        vals2 = ((vals - np.min(vals)) * (1 / m) + plusadd)**2
        simpls2[inds] = vals2

        tmp = sp_A_const_t[:, inds].dot(vals2)

        indices_np = tmp.reshape(-1).argsort()[-(500 + split):][::-1]
        indices_np = indices_np[np.isin(indices_np, p1.song_code) == False][:500]
        dfs.append(pd.DataFrame({
            'pid': np.repeat(pid_, 500),
            'pos': np.arange(500),
            'song_id': indices_np,
            'score': tmp[indices_np, 0]
        }))
    recdf = pd.concat(dfs, axis=0)
    recdf['pid'] = reverse_code(recdf['pid'], pid_codes)
    recdf['song_id'] = reverse_code(recdf['song_id'], song_codes)
    return (recdf, ndcgs)


pool = ProcessPoolExecutor(threads)
res = list(pool.map(recs_for_ids, np.array_split(test_agg.pid_code.unique(), threads)))
pool.shutdown()

recdf = pd.concat([r[0] for r in res], axis=0)
recdf.to_csv('output/%s.csv' % "-".join(test_task), index=False)
