import pandas as pd
all_tasks = [['0t'], ['1t'], ['5t', '5nt'], ['10t', '10nt'], ['25f'], ['25r'], ['100f'], ['100r']]

dfs = [pd.read_csv('output/%s.csv' % ('-'.join(task))) for task in all_tasks]
lists = pd.concat(dfs, axis=0)

print(lists.pid.value_counts().value_counts())

lists.to_csv('output/merged.csv')