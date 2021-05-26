import numpy
import time


def multiply(arr1, arr2):
    result = numpy.zeros((len(arr1), 2))
    return arr1 * arr2

n_pixels = 50000000
x = numpy.random.random(n_pixels)
y = numpy.random.random(n_pixels)

tic = time.time()
z = x + y
toc = time.time()
print(toc-tic)

tic = time.time()
z = x * y
toc = time.time()
print(toc-tic)

tic = time.time()
z = multiply(x, y)
toc = time.time()
print(toc-tic)