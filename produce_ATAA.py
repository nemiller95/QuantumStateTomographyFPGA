import sys
import numpy as np
import cmath
import timeit
import time
import itertools
np.set_printoptions(threshold=sys.maxsize)

def produce_vectors(ataat,n):
	binwidth = 2*n
	width = binwidth+2
	nwidth = 2**n
	depth = 4**n
	indices_r = np.array([],dtype='S'+str(width*nwidth))
	rowind=0
	for i in range(len(ataat)):
		for j in range(len(ataat)):
			if(ataat[i][j]==0):
				continue
			elif(ataat[i][j]==1):
				try:
					indices_r[rowind] = indices_r[rowind]+'00'+str(f'{j:0{binwidth}b}') # 00 = +
				except:
					indices_r = np.append(indices_r,'00'+str(f'{j:0{binwidth}b}'))
			elif(ataat[i][j]==-1):
				try:
					indices_r[rowind] = indices_r[rowind]+'01'+str(f'{j:0{binwidth}b}') # 01 = - //'{0:04b}'.format(j)
				except:
					indices_r = np.append(indices_r,'01'+str(f'{j:0{binwidth}b}'))
			elif(ataat[i][j]==1j):
				try:
					indices_r[rowind] = indices_r[rowind]+'10'+str(f'{j:0{binwidth}b}') # 10 = +j
				except:
					indices_r = np.append(indices_r,'10'+str(f'{j:0{binwidth}b}'))
			elif(ataat[i][j]==-1j):
				try:
					indices_r[rowind] = indices_r[rowind]+'11'+str(f'{j:0{binwidth}b}') # 11 = -j
				except:
					indices_r = np.append(indices_r,'11'+str(f'{j:0{binwidth}b}'))
		rowind = rowind+1
	return indices_r

def produce_vectors_from_tensor(encoded,n):
	binwidth = 2*n
	width = binwidth+2
	nwidth = 2**n
	depth = 4**n
	indices = ["" for x in range(depth)]
	rowind=0
	for i in range(len(encoded)):#
		if (encoded[i][2] == 1):
			try:
				indices[int(encoded[i][0])] = str(indices[int(encoded[i][0])])+'00'+str(f'{int(encoded[i][1]):0{binwidth}b}')
			except:
				indices[int(encoded[i][0])] = str('00'+str(f'{int(encoded[i][1]):0{binwidth}b}'))
		elif (encoded[i][2] == -1):
			try:
				indices[int(encoded[i][0])] += '01'+str(f'{int(encoded[i][1]):0{binwidth}b}')
			except:
				indices[int(encoded[i][0])] = str('01'+str(f'{int(encoded[i][1]):0{binwidth}b}'))
		elif (encoded[i][2] == 1j):
			try:
				indices[int(encoded[i][0])] += '10'+str(f'{int(encoded[i][1]):0{binwidth}b}')
			except:
				indices[int(encoded[i][0])] = str('10'+str(f'{int(encoded[i][1]):0{binwidth}b}'))
		elif (encoded[i][2] == -1j):
			try:
				indices[int(encoded[i][0])] += '11'+str(f'{int(encoded[i][1]):0{binwidth}b}')
			except:
				indices[int(encoded[i][0])] = str('11'+str(f'{int(encoded[i][1]):0{binwidth}b}'))
	indices_np = np.array(indices)
	return indices_np

def produce_tensor(n):
	m = 1. + 0.j #complex(1,0)
	jm = 0. + 1.j #complex(0,1)
	x = 2**(n) + 2
	tensor1 = np.zeros((8,3), dtype=complex)
	tensor1 = np.array([[0,0,m], [0,3,m], [1,1,m], [1,2,jm], [x-1, 0, m], [x-2,1,m], [x-2, 2, -jm], [x-1,3,-m]])

	for i in range(1, n):
		B11 = np.hstack((tensor1[:, :1], tensor1[:, 1:2], tensor1[:, 2:])) 
		B22 = np.hstack((tensor1[:, :1] + int(2**(i))*np.ones((8**i, 1)), tensor1[:, 1:2] + 4**i*np.ones((8**i, 1)), tensor1[:, 2:]))
		B23 = np.hstack((tensor1[:, :1] + int(2**(i))*np.ones((8**i, 1)), tensor1[:, 1:2] + 2*4**i*np.ones((8**i, 1)), jm*tensor1[:, 2:]))
		B14 = np.hstack((tensor1[:, :1], tensor1[:, 1:2] + 3*4**i*np.ones((8**i, 1)), tensor1[:, 2:]))
		B41 = np.hstack((tensor1[:, :1] + int(x + 2**n)*np.ones((8**i, 1)), tensor1[:, 1:2], tensor1[:, 2:]))
		B32 = np.hstack((tensor1[:, :1] + int(x + 2**n-2**(i))*np.ones((8**i, 1)), tensor1[:, 1:2] + 4**i*np.ones((8**i, 1)), tensor1[:, 2:]))
		B33 = np.hstack((tensor1[:, :1] + int(x + 2**n-2**(i))*np.ones((8**i, 1)), tensor1[:, 1:2] + 2*4**i*np.ones((8**i, 1)), -jm*tensor1[:, 2:]))
		B44 = np.hstack((tensor1[:, :1] + int(x + 2**n)*np.ones((8**i, 1)), tensor1[:, 1:2] + 3*4**i*np.ones((8**i, 1)), -1*tensor1[:, 2:]))
		x = 2*x + 2**n
		tensor1 = np.vstack((B11, B14, B41, B44, B22, B32, B23, B33))
	return tensor1

def pauli_nqubits(a_vector):
    multi_q_gate = np.array([1])
    for a in a_vector: #remove the reversed for straightforward
        if(a==3):
            multi_q_gate = np.kron(multi_q_gate,pauli_z)
        elif(a==2):
            multi_q_gate = np.kron(multi_q_gate,pauli_y)
        elif(a==1):
            multi_q_gate = np.kron(multi_q_gate,pauli_x)
        elif(a==0):
            multi_q_gate = np.kron(multi_q_gate,pauli_i)
    return multi_q_gate

def gen_all_avecs(size):
    iterable_wI = np.array([0,1,2,3])
    with_Is = list(itertools.product(iterable_wI, repeat=size))
    return with_Is


pauli_i = np.array([[1,0],[0,1]])
pauli_x = np.array([[0,1],[1,0]])
pauli_y = np.array([[0,-1j],[1j,0]])
pauli_z = np.array([[1,0],[0,-1]])



for n in range(1,3):
  # time
  tensor_start = time.time_ns()
  tens = produce_tensor(n)
  print(tens.shape)
  vecten = produce_vectors_from_tensor(tens,n)
  print("Tensor Pattern Time: " + str((time.time_ns() - tensor_start)/(10**9)))

  #time
  long_start = time.time_ns()
  # create ataat
  all_avec = gen_all_avecs(n)
  basis_matrix = np.array([])
  for avec in all_avec:
    sigma_i = pauli_nqubits(avec)
    try:
      basis_matrix = np.vstack((basis_matrix, np.ndarray.flatten(sigma_i)))
    except:
      basis_matrix = np.ndarray.flatten(sigma_i)
  basis_done = time.time_ns()
  atainv = np.linalg.inv(np.matmul(np.transpose(basis_matrix),basis_matrix)).real
  atainva = np.matmul(atainv,np.transpose(basis_matrix))*(2**n)
  veclong = produce_vectors(atainva,n)
  print("Long Way Time: " + str((time.time_ns() - long_start)/(10**9)))
  print("Not Incl Basis: " + str((time.time_ns() - basis_done)/(10**9)))

  #print(vecten)
  #print(veclong)
  print(veclong.shape)
  print(vecten.shape)

  # compare times and equality
  for i in range(len(vecten)):
    for j in range(len(vecten[0])):
      if (veclong[i][j] != vecten[i][j]):
        print("Mismatch " + str(n))
