import pickle as pkl 
import os
import sys
sys.path.append("..")

def merge():
    short_root = '../data/FinalData/ShortCodes'
    long_root = '../data/FinalData/LongCodes'
    all_root = '../data/FinalData/AllCodes'

    names = ['codes.pkl', 'graphs.pkl', 'names.pkl']

    for name in names:
        with open(os.path.join(short_root, name), 'rb') as f:
            short = pkl.load(f)
        with open(os.path.join(long_root, name), 'rb') as f:
            long = pkl.load(f)
        all_data = short + long
        with open(os.path.join(all_root, name), 'wb') as f:
            pkl.dump(all_data, f)
    
merge()
    
