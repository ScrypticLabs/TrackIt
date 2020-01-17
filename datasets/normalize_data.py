import os
import numpy as np

def normalize_ndarray(data_type, resolution):
    X = np.load(data_type+"/"+resolution+"/X_"+data_type+".npy") 
    Y = np.load(data_type+"/"+resolution+"/Y_"+data_type+".npy")

    X = X / 255.0
    Y = Y / 255.0

    np.save(data_type+"/"+resolution+"/X_"+data_type+"_normalized.npy", X)
    np.save(data_type+"/"+resolution+"/Y_"+data_type+"_normalized.npy", Y)

def normalize_ndarray_by_ball(path):
    X = np.load(path) 
    X = X / 255.0
    np.save(path.split(".")[0]+"_normalized.npy", X)

# normalize_ndarray("train", "32")
# normalize_ndarray("val", "32")
# normalize_ndarray("test", "32")

# normalize_ndarray_by_ball("data/32/1.npy")
# normalize_ndarray_by_ball("data/32/2.npy")
# normalize_ndarray_by_ball("data/32/3.npy")
# normalize_ndarray_by_ball("data/32/4.npy")
# normalize_ndarray_by_ball("data/32/5.npy")