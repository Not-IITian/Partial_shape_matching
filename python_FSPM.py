import tensorflow as tf
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.solvers import ConjugateGradient
import numpy as np
import scipy.io as sio
import h5py

def fro_norm(tensor):
    square_tensor = tf.square(tensor)
    tensor_sum = tf.reduce_sum(square_tensor)
    return tensor_sum

data_1 = './Data/cat0_M.mat'
data_2 = './Data/cat0_part1.mat'
k = 90
part = sio.loadmat(data_2)
M = sio.loadmat(data_1)

#print(part)
#part = parts{1};
fullshape_idx = part['fullshape_idx']

S_part = part['S'].todense()
evecs_part = part['evecs'][:,0:k]
part_s = M['shot']
part_sh = part_s[fullshape_idx,:]
evecs_T = np.transpose(evecs_part)
evec_trans = tf.matmul(evecs_T, S_part)

part_sh = np.squeeze(part_sh)
A = tf.matmul(evec_trans,part_sh)

#evecs_M = M['evecs']
#evecs_M = evecs_M

evec_trans_M = tf.matmul(tf.transpose(M['evecs'][:,0:k]),M['S'].todense())
B = tf.matmul(evec_trans_M, M['shot'])

#dummy = part['evals'] - max(M['evals'])
#est_rank = tf.reduce_sum(dummy<0) # no of eigen vallues strictly geater than part

est_rank = 36;

manifold = Stiefel(k, est_rank)
M_evals = M['evals'][0:k]

lambda1 = tf.diag(tf.squeeze(M_evals))
mu = tf.constant(1e-1, dtype=tf.float64)
 # off-diagonal mask
W = 1 - tf.diag(tf.ones(est_rank,1))
W =tf.cast(W, dtype = tf.float64)
rnk = est_rank
rnk2 = k-rnk
v2 = tf.zeros((rnk,rnk2))
v1 = tf.eye(rnk)

C = tf.concat(values=[v1, v2], axis = 1)
C  =tf.cast(C, dtype = tf.float64)
X = tf.Variable(tf.placeholder(tf.float64,shape=(k, est_rank)))

pre_mul = tf.matmul(tf.matmul(tf.transpose(X),lambda1),X)

dummy = tf.matmul(C,A )-tf.matmul(tf.transpose(X),B)

cost= mu*fro_norm(dummy) + fro_norm(tf.multiply(pre_mul,W))
#+ sum(sum( (lambda2 .*W).^2 ))

problem = Problem(manifold=manifold, cost=cost, arg=X)

# (3) Instantiate a Pymanopt solver

#check if you can insert x0 as init point..
#x= tf.concat(values=[ tf.eye(rnk), tf.zeros((rnk2,rnk))], axis = 0)
#x = np.concatenate((np.eye(rnk),np.zeros((rnk2,rnk))), axis=0)
#x = tf.cast(x0, dtype = tf.float64)
solver = ConjugateGradient( maxiter=20000)

# let Pymanopt do the rest
Xopt = solver.solve(problem)

print(Xopt)


