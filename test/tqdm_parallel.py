"""
Testing multiprocessing + tqdm
https://stackoverflow.com/q/41920124/2229219
"""
from multiprocessing import cpu_count, freeze_support
from multiprocessing import Pool
from tqdm import tqdm

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Test multiprocessing stuff.")
    parser.add_argument("-n", "--number", help="number of iterations", type=int, default=7000000)
    args = parser.parse_args()
    return args

def _foo(my_number):
    square = my_number * my_number
    return square, square

if __name__ == '__main__':
    freeze_support()
    print(f"CPU count: {cpu_count()}")

    args = parse_args()

    with Pool() as p:
        r1, r2 = zip(*tqdm(p.imap(_foo, range(args.number), chunksize=500), total=args.number))
