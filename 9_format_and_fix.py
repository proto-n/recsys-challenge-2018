import pandas as pd
import numpy as np
import gzip
import sys

if(len(sys.argv) != 3):
    print(
"""################
Submission fixing formatting script.
Usage:
    python format_and_fix_submission.py [submission.csv] [outfile.csv]
Here submission.csv is a flat-style recommendation list, i.e. with columns "pid,song_id,pos".
################""")
    quit()

infile = sys.argv[1]
outfile = sys.argv[2]

print("reading files")
pids = pd.read_csv('data/challenge_set/playlist_meta.csv', usecols=['pid']).pid.values
playlists = pd.read_csv('data/million_playlist_dataset/playlists.csv')
challenge_set = pd.read_csv('data/challenge_set/playlists.csv')
top1000 = pd.DataFrame({'song_id':playlists.song_id.value_counts().iloc[:1000].index})
prev_song_meta = pd.read_csv('data/million_playlist_dataset/song_meta_no_duplicates.csv', usecols=['track_uri', 'song_id'])
song_id_recode = dict(prev_song_meta[['song_id', 'track_uri']].values)

submission = pd.read_csv(infile)

print("searching for duplicates...", end=" ")
origlen = len(submission)
submission.drop_duplicates(inplace=True)
droppedlen = len(submission)
if(origlen != droppedlen):
    print("%d duplicates found, dropping them" % (origlen - droppedlen))

print("searching for invalid recommendations...", end=" ")
submision_joined = submission[['pid', 'song_id']].join(challenge_set.drop_duplicates(subset=['pid', 'song_id']).set_index(['pid', 'song_id']), on=['pid', 'song_id'], rsuffix='_r')
submission = submission[submision_joined.pos.isnull()]
filteredlen = len(submission)
if(filteredlen != droppedlen):
    print("%d invalid recommendations found, dropping them" % (droppedlen - filteredlen))

submission['song_id'] = submission['song_id'].map(song_id_recode)
top1000['song_id'] = top1000['song_id'].map(song_id_recode)
challenge_set['song_id'] = challenge_set['song_id'].map(song_id_recode)

print("filling missing entries from popularity...", end=" ")
playlist_groups = submission.groupby('pid')
seed_groups = challenge_set.groupby('pid')

corrected_submissions = []
extensions = 0
for i in pids:
    songs = []
    try:
        g = playlist_groups.get_group(i)
        if(len(g) < 500):
            extens = top1000[~top1000.song_id.isin(g.song_id)]
            if(i in seed_groups.groups):
                extens = extens[~extens.song_id.isin(seed_groups.get_group(i).song_id)]
            songs = pd.concat([g.song_id, extens.song_id.head(500 - len(g))])
            extensions += 500 - len(g)
        else:
            songs = g.song_id
    except:
        extensions += 500
        songs = top1000.song_id.head(500)
    if len(songs) != 500:
        print(i)
    corrected_submissions.append([str(i)] + list(songs.values))
print("extended with %d songs in total" % extensions)

print("writing csv file")
with open(outfile, "w") as f:
    f.write('team_info,Definitive Turtles,main,kdomokos@info.ilab.sztaki.hu\n\n')
    for l in corrected_submissions:
        f.write(", ".join(l) + "\n")

print("zipping...", end=" ")
f_in = open(outfile, "rb")
f_out = gzip.open(outfile + ".gz", 'wb')
f_out.writelines(f_in)
f_out.close()
f_in.close()
print(outfile + ".gz ready")
