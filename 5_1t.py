import pandas as pd
import numpy as np
import scipy.sparse as spl
from concurrent.futures import ProcessPoolExecutor
import sys

sys.path.append('./utils')
from name_normalize import name_normalize

threads = 20
all_tasks = [
    [1, 1500, ['1t'], 0.933],
]

split, knn_k, test_task, alpha = all_tasks[0]


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

test['val'] = 1
train['val'] = 1
train_agg = train.groupby(['pid_code', 'song_code']).agg({'pos': 'min', 'val': 'sum'}).reset_index()
test_agg = test.groupby(['pid_code', 'song_code']).agg({'pos': 'min', 'val': 'sum'}).reset_index()

train_agg['val_stoch'] = train_agg.groupby('pid_code').val.transform(lambda x: x / np.linalg.norm(x))
test_agg['val_stoch'] = test_agg.groupby('pid_code').val.transform(lambda x: x / np.linalg.norm(x))

playlist_meta['name_normalized'] = name_normalize(playlist_meta['name'])
playlist_meta['name_code'], name_codes = recode(playlist_meta['name_normalized'])


sp_A = spl.coo_matrix((train_agg['val_stoch'].values.T, train_agg[['pid_code', 'song_code']].values.T))
sp_A._shape = (int(playlist_meta.pid_code.max() + 1), int(song_meta.song_code.max() + 1))
sp_A = sp_A.tocsr()
sp_A_t = sp_A.T
sp_A_const = spl.coo_matrix((train_agg['val'].values.T, train_agg[['pid_code', 'song_code']].values.T))
sp_A_const._shape = (int(playlist_meta.pid_code.max() + 1), int(song_meta.song_code.max() + 1))
sp_A_const = sp_A_const.tocsr()
sp_A_const_t = sp_A_const.T

name_code_by_pid_code = dict(zip(playlist_meta['pid_code'], playlist_meta['name_code']))


def recs_for_ids(ids_):
    dfs = []
    ndcgs = []
    for pid_ in ids_:
        p1 = test_agg[test_agg.pid_code == pid_]
        np_p1 = np.zeros([int(song_meta.song_code.max() + 1), 1])
        np_p1[p1.song_code.values] = p1[['val_stoch']].values

        simpls = sp_A.dot(np_p1)

        inds = simpls.reshape(-1).argsort()[-knn_k:][::-1]
        vals = simpls[inds]

        vals2 = vals
        tmp1 = sp_A_const_t[:, inds].dot(vals2)

        name_ = name_code_by_pid_code[pid_]
        simpls2 = (playlist_meta['name_code'] == name_).astype(int).values.reshape((-1, 1))
        inds = np.nonzero(simpls2)[0]
        tmp2 = sp_A_const_t[:, inds].dot(np.ones((len(inds), 1)))**(1 / 2)

        tmp = alpha * tmp1 + (1 - alpha) * tmp2

        indices_np = tmp.reshape(-1).argsort()[-(500 + 1):][::-1]
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
