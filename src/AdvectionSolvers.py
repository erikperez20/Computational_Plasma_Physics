import numpy as np

class Advection_Methods:
	def __init__(self , L , T , a , init_cond , N , M , File_Name):

		''' Initialization of the Constant Coefficient 1D Advection Solver. 
		Inputs:
			- L,T: Size of the container and maximum time of integration
			- a: Advection Coefficient (float number)
			- init_cond: Initial Condition function, depends of the x positions vector
			- N,M: Divisions in the x dimension and in the time dimension respectively
			- File_Name: Name of the file to store solutions (in txt format) '''

		self.L = L # Size of the container 
		self.T = T # Max Time
		
		self.a = a # Advection Coefficient
		self.init_cond = init_cond # Initial condition function, depends of one variable

		self.N = N # Divisions in the x dimension in N slots
		self.M = M # Divisions of time in M slots
		
		self.dt = self.T/self.M # Time step size
		self.h = self.L/self.N # Space step size
		
		self.File_Name = File_Name
	
	def grid_points(self,start_point, grid_size, step_size):
		'''Construct the grid points'''
		return np.arange(start_point,grid_size + step_size,step_size)

class Euler_Upwind_1D(Advection_Methods):
	''' Euler Upwind Scheme implementation for 1D Constant Advection coefficient. It runs for M time steps
		and u is L periodic, i.e. u[0] = u[N] '''

	def run(self):
		
		# Grids:
		x_grid = self.grid_points(0.0,self.L,self.h) # Periodic condition, solve the N first points
		
		# Initial Condition:
		u_new = self.init_cond(x_grid)
		
		# Multiplicative constant
		s = (self.dt/self.h)*self.a
		
		# The Courant–Friedrichs–Lewy condition 
		if abs(s) <= 1:

			# Txt to store values
			a_file = open( self.File_Name + ".txt" , "w")
			np.savetxt(a_file , u_new) # Save initial condition
			
			# Iteration over the time step sizes
			for k in range(1,self.M+1):

				# Replace the new solution by the old and apply periodic condition, solve the N first points (u[0] = u[-1])
				u_old = u_new.copy()[:-1]
				
				# Conditions for periodic boundary conditions
				# Condition 1: a<0
				if max(self.a, 0.0) == 0.0:
					u_forward = np.roll(u_old,-1) # Vectorial Displacement to the right
					u_new = u_old - s*(u_forward - u_old) # Formula for u(t+dt)        

				# Condition 2: a>0
				elif min(self.a,0.0) == 0.0:
					u_backward = np.roll(u_old,1) # Vectorial Displacement to the left
					u_new = u_old - s*(u_old - u_backward) # Formula for u(t+dt)

				u_new = np.append(u_new , u_new[0]) # Periodic Condition u[0] = u[N]
				np.savetxt(a_file, u_new) #Save the calculated array at time k
			
			a_file.close() # close file

		else:
			raise ValueError('The Courant–Friedrichs–Lewy condition must be satisfied, a*dt/dx must be <= 1')

class Lax_Wendroff_1D(Advection_Methods):
	''' Lax-Wendroff Scheme implementation for 1D Constant Advection coefficient. It runs for M time steps 
		and u is L periodic, i.e. u[0] = u[N] '''

	def run(self):
		
		# Grid:
		x_grid = self.grid_points(0.0,self.L,self.h)
		
		# Initial Condition:
		u_new = self.init_cond(x_grid)
		
		# Multiplicative constant
		s = (self.dt/self.h)*self.a
		
		# Courant–Friedrichs–Lewy numerical stability condition
		if abs(s) <= 1:
			
			# Txt to store values
			a_file = open(self.File_Name+".txt", "w")
			np.savetxt(a_file,u_new)
			
			# Iteration over the time step sizes
			for k in range(1,self.M+1):

				# Replace the new solution by the old and apply periodic condition, solve the N first points (u[0] = u[-1])
				u_old = u_new.copy()[:-1]
				
				u_forward = np.roll(u_old,-1) # u_j+1 (Vectorial Displacement to the right)
				u_backward = np.roll(u_old,1) # u_j-1 (Vectorial Displacement to the left)
				
				# Numerical scheme equation in two steps
				term = u_old - (s/2.0) * (u_forward - u_backward)
				u_new =  term + (s**2/2.0) * (u_forward - 2.0*u_old + u_backward)
				
				u_new = np.append(u_new , u_new[0]) # Periodic Condition u[0] = u[N]
				np.savetxt(a_file, u_new) # Save the solution at time step k
				
			a_file.close()
		
		else:
			raise ValueError('The Courant–Friedrichs–Lewy condition must be satisfied, a*dt/dx must be <= 1')