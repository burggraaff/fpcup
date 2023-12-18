# https://stackoverflow.com/a/41921948/2229219
from multiprocessing import Pool
import time
from tqdm import *

def _foo(my_number):
   square = my_number * my_number
   return square

if __name__ == '__main__':
    with Pool(processes=8) as p:
        max_ = 300000
        with tqdm(total=max_) as pbar:
            for _ in p.imap_unordered(_foo, range(0, max_)):
                pbar.update()
