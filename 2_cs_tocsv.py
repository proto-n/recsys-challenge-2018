import sys
import json
import time
import os
import re
from pprint import pprint
from collections import defaultdict
import pandas as pd

path = "data_raw/challenge_set"

filenames = os.listdir(path)


def escapefunc(st):
    return "\"%s\"" % st.replace("\"", "\"\"")


prev_song_meta = pd.read_csv('data/million_playlist_dataset/song_meta_no_duplicates.csv', usecols=['track_uri', 'song_id'])

song_ids = defaultdict(lambda: None, prev_song_meta[['track_uri', 'song_id']].values)
next_song_id = 0


playlist_meta = open("data/challenge_set/playlist_meta.csv", "w")
song_meta = open("data/challenge_set/song_meta.csv", "w")
playlists = open("data/challenge_set/playlists.csv", "w")

playlist_meta.write('pid,name,num_tracks,num_holdouts,num_samples\n')
song_meta.write('song_id,album_name,album_uri,artist_name,artist_uri,duration_ms,track_name,track_uri\n')
playlists.write('pid,song_id,pos\n')


def write(playlist):
    global next_song_id

    pid = playlist['pid']
    name = ""
    if('name' in playlist):
        name = playlist['name']
    playlist_meta.write('%s,%s,%s,%s,%s\n' % (
        playlist['pid'],
        escapefunc(name),
        playlist['num_tracks'],
        playlist['num_holdouts'],
        playlist['num_samples'],
    ))

    for i, track in enumerate(playlist['tracks']):
        song_id = song_ids[track['track_uri']]
        if song_id is None:
            song_id = next_song_id
            song_ids[track['track_uri']] = song_id
            next_song_id += 1

        song_meta.write('%s,%s,%s,%s,%s,%s,%s,%s\n' % (
            song_id,
            escapefunc(track['album_name']),
            track['album_uri'],
            escapefunc(track['artist_name']),
            track['artist_uri'],
            track['duration_ms'],
            escapefunc(track['track_name']),
            track['track_uri'],
        ))

        playlists.write('%s,%s,%s\n' % (pid, song_id, track['pos']))


for i, filename in enumerate(sorted(filenames)):
    print("processing ", i, filename)
    if filename.endswith(".json"):
        fullpath = os.sep.join((path, filename))
        f = open(fullpath)
        js = f.read()
        f.close()
        mpd_slice = json.loads(js)
        for playlist in mpd_slice['playlists']:
            write(playlist)

playlist_meta.close()
song_meta.close()
playlists.close()
