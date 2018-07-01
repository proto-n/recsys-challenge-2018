import pandas as pd
import numpy as np
import re


def name_normalize(name_series):
    def f(x): return re.sub('[-\.\$\+>#\}\{\*<&@%;\\\)\(="?!\[\]~\^/:,_\|\s–—]+', '.', str(x).lower().replace("'", "")).strip(".")

    def find(x): return re.match('^[\w\d\.]+$', x) == None
    name_series = name_series.apply(f)
    smileys = name_series[name_series.apply(find)]

    def f(x): return "".join(set(x))
    smileys = smileys.apply(f)
    name_series.loc[smileys.index] = smileys
    return name_series
