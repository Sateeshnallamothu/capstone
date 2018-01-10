# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 14:39:52 2017

@author: SateeshSwathi
"""

import pickle
import os
import sys

os.chdir('C:\\Users\\SateeshSwathi\\Documents\\Python Scripts\\')

####  load data
dictionary,idx2word,prob_table_kn2,prob_table_bi,dist_ugs = pickle.load(open("ngramdict.pickle","rb"))
#gc.collect()
def pred_next(w1,w2):
    idx1 = dictionary[w1] if dictionary.get(w1) is not None else 0
    idx2 = dictionary[w2] if dictionary.get(w2) is not None else 0 
    rs = []
    if (idx1 > 0 and idx2 > 0 and (prob_table_kn2.get(idx1,idx2) is not None)):
        for tkn,prob in prob_table_kn2[idx1,idx2]:
            rs.append((idx2word[tkn],prob))
    elif ((idx2 > 0) and (prob_table_bi.get(idx2) is not None)):
        for tkn,prob in prob_table_bi[idx2]:
            rs.append((idx2word[tkn],prob))
    else:
        rs.append(idx2word[0])
    return rs

if __name__ == '__main__' :    
    w1,w2 = sys.argv[1],sys.argv[2]
    print(pred_next(w1,w2))