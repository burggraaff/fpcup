"""
Testing multiprocessing + tqdm
https://stackoverflow.com/q/41920124/2229219
"""

from multiprocessing import cpu_count
from multiprocessing import Pool
from tqdm import tqdm
import time

print(f"CPU count: {cpu_count()}")

def _foo(my_number):
    square = my_number * my_number
    return square, square

if __name__ == '__main__':
    nmax = 7000000
    with Pool() as p:
        r1, r2 = zip(*tqdm(p.imap(_foo, range(nmax), chunksize=5), total=nmax))
