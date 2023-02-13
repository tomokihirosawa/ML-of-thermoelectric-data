import torch
import time

# define additional functions here

# check run times of indiviual steps
class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))

if __name__ == '__main__':
    # check additional helper functions that you need
    print('Check utils')
