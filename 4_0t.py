import pandas as pd
import numpy as np
from utils.name_normalize import name_normalize

alldata = pd.read_csv('data/million_playlist_dataset/playlists.csv')

meta = pd.read_csv('data/million_playlist_dataset/song_meta_no_duplicates.csv')
playlist_meta = pd.read_csv('data/million_playlist_dataset/playlist_meta.csv')
playlist_meta_challenge = pd.read_csv('data/challenge_set/playlist_meta.csv')
playlist_meta = pd.concat([playlist_meta[['pid', 'name']], playlist_meta_challenge[['pid', 'name']]], axis=0, ignore_index=True)

playlist_meta['name_normalized'] = name_normalize(playlist_meta['name'])
alldata = alldata.join(playlist_meta.set_index('pid')['name_normalized'], on='pid')

song_counts = alldata.groupby(['name_normalized', 'song_id']).pid.count().rename('count').reset_index()
song_counts['confidence'] = np.log(song_counts['count'] / song_counts.groupby('name_normalized')['count'].transform(np.sum))


tasks = pd.read_csv('data/challenge_set/playlist_meta_tasks.csv')

recs = pd.DataFrame({'pid': tasks[tasks['task'] == '0t'].pid.values})
recs = recs.join(playlist_meta.set_index('pid')['name_normalized'], on='pid')
recs_sc = recs.join(
    song_counts
    .set_index('name_normalized')[['song_id', 'confidence']]
    .rename(columns={'confidence': 'song_confidence'}),
    on='name_normalized'
)
recs_sc.dropna(subset=['song_id'], inplace=True)

recs_sc.sort_values(['pid', 'song_confidence'], ascending=False, inplace=True)
recs_sc['pos'] = 1
recs_sc['pos'] = recs_sc.groupby('pid').pos.transform(np.cumsum)
rec_song = recs_sc[recs_sc['pos'] <= 500].copy()
rec_song['song_id'] = rec_song['song_id'].astype(np.int)
rec_song[['pid', 'song_id', 'pos']].to_csv('output/0t.csv', index=False)
