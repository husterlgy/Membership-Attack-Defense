# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 10:01:16 2018

@author: husterlgy
"""

n_hidden=50
batch_size=100
epochs=100
learning_rate=0.01
model='nn'
l2_ratio=1e-7
rtn_layer=True


n_hidden=50
epochs=100
n_shadow=20
learning_rate=0.05 
batch_size=100 
l2_ratio=1e-7
model='nn' 
save=True






import numpy as np  
import time  
import theano  
A = np.random.rand(1000,10000).astype(theano.config.floatX)  
B = np.random.rand(10000,1000).astype(theano.config.floatX)  
np_start = time.time()  
AB = A.dot(B)  
np_end = time.time()  
X,Y = theano.tensor.matrices('XY')  
mf = theano.function([X,Y],X.dot(Y))  
t_start = time.time()  
tAB = mf(A,B)  
t_end = time.time()  
print "NP time: %f[s], theano time: %f[s] (times should be close when run on CPU!)" %(  
                                           np_end-np_start, t_end-t_start)  
print "Result difference: %f" % (np.abs(AB-tAB).max(), ) 








from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], T.exp(x))
print (f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print ('Looping %d times took' % iters, t1 - t0, 'seconds')
print ('Result is', r)
if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
    print ('Used the cpu')
else:
    print ('Used the gpu')























