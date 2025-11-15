import os
import time
import numpy as np
from tqdm import tqdm
from pybloom_live import BloomFilter
import argparse


n_gram = 8
capacity = 3e10
error_rate = 0.0001


def bulid_filter(data, bf_path, interval, n_gram=n_gram, capacity=capacity, error_rate=error_rate):
    flag = time.time()
    bloom_filter = BloomFilter(capacity=capacity, error_rate=error_rate)
    print('filter built: %.4f'%((time.time()-flag)/60))
    for i in range(len(data)-n_gram):
        bloom_filter.add(data[i:i+n_gram])
        if i % interval == 0:
            print('%d\t%.4f'%(i, (time.time()-flag)/60))
    flag = time.time()
    print(bf_path, type(bf_path))
    bloom_filter.tofile(open(bf_path, 'wb'))
    print('filter saved: %.4f'%((time.time()-flag)/60))


def read_filter(bf_path):
    flag = time.time()
    if type(bf_path) == str:
        bloom_filter = BloomFilter.fromfile(open(bf_path, 'rb'))
        print('filter read: %.4f'%((time.time()-flag)/60))

    elif type(bf_path) == list:
        bloom_filter = BloomFilter.fromfile(open(bf_path[0], 'rb'))
        print('filter read: %.4f'%((time.time()-flag)/60))
        flag = time.time()
        for dp in bf_path[1:]:
            bloom_filter_temp = BloomFilter.fromfile(open(dp, 'rb'))
            bloom_filter = bloom_filter.union(bloom_filter_temp)
            print('filter read: %.4f'%((time.time()-flag)/60))
            flag = time.time()
    
    return bloom_filter


def union_filter(bf_path, to_path):
    flag = time.time()
    bloom_filter = BloomFilter.fromfile(open(bf_path[0], 'rb'))
    print('filter read: %.4f'%((time.time()-flag)/60))

    flag = time.time()
    for dp in bf_path[1:]:
        bloom_filter_temp = BloomFilter.fromfile(open(dp, 'rb'))
        bloom_filter = bloom_filter.union(bloom_filter_temp)
        print('filter read: %.4f'%((time.time()-flag)/60))
        flag = time.time()

    bloom_filter.tofile(open(to_path, 'wb'))
    print('filter saved: %.4f'%((time.time()-flag)/60))


def checkDup(data, bloom_filter, n_gram=n_gram, interval=1000000):
    dup = 0
    flag = time.time()
    for i in range(len(data)-n_gram):
        if data[i:i+n_gram] in bloom_filter:
            dup += 1
        if i % interval == 0:
            print('%d\t%.4f'%(i, (time.time()-flag)/60))
    return dup


if __name__ == '__main__':
    1

    # bloom_filter = read_filter(bf_path[0])
    # dup = checkDup(data[1000000:2000000], bloom_filter)
    # print(dup)  # 33020

    # bulid_filter(data[1000000:2000000], bf_path[1])

    # bloom_filter = read_filter(bf_path)
    # dup = checkDup(data[2000000:3000000], bloom_filter)
    # print(dup) 

    # union_filter(bf_path, to_path)

    # bulid_filter(data[2000000:3000000], bf_path[2])

    # bloom_filter = read_filter(bf_path)
    # dup = checkDup(data[3000000:4000000], bloom_filter)
    # print(dup)  # 40105

    # union_filter(bf_path, to_path)