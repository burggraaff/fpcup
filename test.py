"""
Testing multiprocessing + tqdm
https://stackoverflow.com/q/41920124/2229219
"""

from multiprocessing.dummy import Pool
import tqdm
import time

def _foo(my_number):
    square = my_number * my_number
    return square

if __name__ == '__main__':
    max = 3000000
    with Pool() as p:
        r = list(tqdm.tqdm(p.map(_foo, range(max)), total=max))
