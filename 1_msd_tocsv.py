import sys
import json
import time
import os
import re
from pprint import pprint
from collections import defaultdict
import pandas as pd

path = "data_raw/million_playlist_dataset"

filenames = os.listdir(path)


def escapefunc(st):
    return "\"%s\"" % st.replace("\"", "\"\"")


song_ids = defaultdict(lambda: None)
next_song_id = 0

playlist_meta = open("data/million_playlist_dataset/playlist_meta.csv", "w")
song_meta = open("data/million_playlist_dataset/song_meta.csv", "w")
playlists = open("data/million_playlist_dataset/playlists.csv", "w")

playlist_meta.write('pid,collaborative,duration_ms,modified_at,name,num_albums,num_artist,num_edits,num_followers,num_tracks\n')
song_meta.write('song_id,album_name,album_uri,artist_name,artist_uri,duration_ms,track_name,track_uri\n')
playlists.write('pid,song_id,pos\n')


def write(playlist):
    global next_song_id

    pid = playlist['pid']
    playlist_meta.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (
        playlist['pid'],
        playlist['collaborative'],
        playlist['duration_ms'],
        playlist['modified_at'],
        escapefunc(playlist['name']),
        playlist['num_albums'],
        playlist['num_artists'],
        playlist['num_edits'],
        playlist['num_followers'],
        playlist['num_tracks'],
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
    if filename.startswith("mpd.slice.") and filename.endswith(".json"):
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

song_meta = pd.read_csv('data/million_playlist_dataset/song_meta.csv')
song_meta.drop_duplicates()
song_meta.to_csv('data/million_playlist_dataset/song_meta_no_duplicates.csv')