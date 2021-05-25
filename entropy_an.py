# encoding=utf8 
# The entropy is calculated from the MD simulations
# written by dongshengchen in 7/5/2021
'''ref
[1] J. Chem. Phys. 115, 6289 (2001)
[2] Self-synchronization of thermal phonons at equilibrium (https://arxiv.org/abs/2005.06711)
'''
import numpy as np
# import math
import matplotlib.pyplot as plt
import time as ti
# from scipy import linalg

# np.set_printoptions(threshold=100000000)
# constants
kb = 1.3806504e-23 # J/K Boltzmann
h_rpc = 1.05457266e-34 # J·s, reduced Planck constant
temperature = 300 # K
e = 2.7182818284590452353602874713527 # Euler’s Number

NA = 6.0221415e23 #avogadro constant

a = kb*temperature*(e**2)/(h_rpc**2)

# print(a)

class Entropy(object):
	def __init__(self,a,inter_step,time_step):
		super(Entropy, self).__init__()
		self.a = a
		self.inter_step = inter_step
		self.inter_time = inter_step*time_step*(1e-3) #step-->fs-->ps
		
	def entropy(self,dump,atom_number,sample_inter,position=3):
		data_list = []
		with open(dump,'r') as data:
			for index, line in enumerate(data,1):
				line = line.strip().split()
				# print(line,len(line))
				if len(line) == 6 and 'pp' not in line:
					# print(line)
					data_list.append(line)
		# print(data_list)
		data_array = np.array(data_list).astype(float)

		# print(data_array.shape)

		x = data_array[:,position].reshape(atom_number,-1)*(1e-9) #Angstrom-->m
		m, n = x.shape
		print(x.shape)
		print('Atom number = ',m,'\nLAMMPS Run Step number =',n*self.inter_step,
			'\nTotal Run Time = ',n*self.inter_time,'ps')
		# print(x)
		# sigma = np.cov(x)
		'''covariance matrix of the coordinate fluctuations'''
		sigma_list = []
		for i in range(m):
			for j in range(m):
				xi_average = np.mean(x[i,:])
				xj_average = np.mean(x[j,:])
				sigma = (x[i,:]-xi_average)*(x[j,:]-xj_average)
				sigma_list.append(sigma)
		# print(sigma_list)
		sigma_array = np.array(sigma_list).reshape(m,m,n)
		# print(sigma_array.shape)
		# print(sigma_array[:,:,1].shape)

		'''sampling average from the covariance matrix'''
		sigma_inter = []
		for t in range(n):
			r = int(n/sample_inter)
			if t<r:
				o = t*sample_inter
				p = (t+1)*sample_inter
				sigma_average = sigma_array[:,:,o:p].mean(2)
				# print(sigma_average.shape)
				sigma_inter.append(sigma_average.tolist())
		sigma_inter_array = np.array(sigma_inter)
		print('Average interval of the position x,y,z = ',self.inter_time,'ps')
		print('Average interval of the the covariances = ',self.inter_time*sample_inter,'ps')
		# print(sigma_inter_array.shape)
		# unity matrix
		unity_matrix = np.eye(m)
		# print(unity_matrix)
		# mass matrix
		mass = data_array[:,2].reshape(atom_number,-1)[:,0].reshape(m,1)*(1e-3) # g/mol-->kg/mol
		mass = np.sqrt(mass*mass.T)
		# print(mass.shape,mass)
		entropy_list = []
		for i in range(r):

			b = self.a*(mass*sigma_inter_array[i,:,:])+unity_matrix
			# print(b)
			sign, logdet = np.linalg.slogdet(b)
			# print(sign, logdet)
			## logdet = math.log(linalg.det(b))

			S = 0.5*kb*logdet
			# print(S)
			entropy_list.append(S)
		# print(entropy_list)


		time = n*self.inter_time*np.linspace(0,1,r).reshape(r,1) #ps
		entropy_list = np.array(entropy_list).reshape(r,1)*(NA/atom_number) #J/K-->J/K/mol
				
		print('-----------------End!-----------------')
		return  time, entropy_list


	def plot(self,time,entropy):
		# plot
		plt.scatter(time,entropy)
		plt.xlabel('Time (ps)')
		plt.ylabel('Entropy (J/K-mol)')
		# plt.xlim(-0.1,)
		# plt.ylim(220,240)
		plt.savefig('entropy_an_'+str(temperature)+'.png',dpi=300)
		# plt.show()
		return

# --------VARIABLES-------- #

# the dump file format is "id type mass x y z "
dump = './aniondump_'+str(temperature)+'.dat'

# dump interval step number
inter_step = 5
# sample interval of covariances, 
sample_inter = 100
time_step = 1 #fs
atom_number = 285
# --------MAIN PROGRAM-------- #

if __name__ == '__main__':
	start = ti.time()
	print('-----------------Start!-----------------')
	S = Entropy(a,inter_step,time_step)

	time, entropy_list_x= S.entropy(dump,atom_number,sample_inter,position=3)
	time, entropy_list_y= S.entropy(dump,atom_number,sample_inter,position=4)
	time, entropy_list_z= S.entropy(dump,atom_number,sample_inter,position=5)

	entropy = entropy_list_x + entropy_list_y + entropy_list_z
	
	data = np.hstack((time,entropy))
	entropy_average = np.mean(entropy,axis=0)
	np.savetxt('entropy_an_'+str(temperature)+'.dat',data,'%f ' ' %f')
	np.savetxt('entropy_an_average_'+str(temperature)+'.dat',entropy_average,'%f')
	print(entropy_average)
	S.plot(time,entropy)

print('-----------------Done!-----------------')
end = ti.time()
print('--THIS PROGRAM TOTAL RUNING TIME =',(end-start),' s --')
