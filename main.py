from censor import Censor
import os
import glob
import sys
import time
import math
import multiprocessing
from decouple import config

def censor(image_paths, dest_dir, num_threads):
    c = Censor(dest_dir, num_threads)
    c.censor_files(image_paths)
    c.end()

if __name__ == "__main__":
    args = dict(enumerate(sys.argv))
    img_dir = args.get(1, "images")
    dest_dir = args.get(2, "censored")
    
    NUM_PROCESSES = config("NUM_PROCESSES", default=4, cast=int)
    NUM_THREADS = config("NUM_THREADS", default=2, cast=int)

    print("Starting censoring with {} processes with {} threads each".format(NUM_PROCESSES, NUM_THREADS))

    start_time = time.time()

    data_path = os.path.join(img_dir,'*g')
    images = glob.glob(data_path)
    minsize = 20
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # split into batches
    num_per_process = math.ceil(len(images) / NUM_PROCESSES)
    batches = [[]]

    for image in images:
        batches[-1].append(image)
        if len(batches[-1]) == num_per_process:
            batches.append([])

    if len(batches[-1]) == 0:
        batches.pop()

    proccesses = []

    for i, b in enumerate(batches):
        x = multiprocessing.Process(target=censor, args=(b, dest_dir, NUM_THREADS))
        x.start()
        proccesses.append(x)

    for t in proccesses:
        t.join()

    print("Total time for {} images: {}".format(len(images), time.time() - start_time))

        
