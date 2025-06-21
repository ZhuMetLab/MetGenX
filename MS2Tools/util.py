# !/usr/bin/env python
# -*-coding:utf-8 -*-

def SpecQuery(database, SpecName):
    cur_spec = database.cursor()
    cur_spec.execute("SELECT * FROM " + SpecName)
    return (cur_spec.fetchall())

import numpy as np
import math
def CalSpecVec(spec, intensity, model, normalized=False):
    score = []
    for peak, peak_int in zip(spec, intensity):
        if peak_int == 2:
            peak_int = 0
        if peak in model.wv:
            score.append(model.wv[peak]*math.sqrt(peak_int))
        else:
            score.append(np.zeros(512) * math.sqrt(peak_int))
    score = np.sum(score, axis=0)
    if normalized:
        l2_norm = np.linalg.norm(score)
        score = score / l2_norm
    return score
