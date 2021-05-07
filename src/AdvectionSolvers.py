import numpy as np
from scipy import integrate

class Advection_Methods_1D:
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

class Euler_Upwind_1D(Advection_Methods_1D):
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

class Lax_Wendroff_1D(Advection_Methods_1D):
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

class Spectral_Method_1D(Advection_Methods_1D):
	''' Spectral Scheme implementation for 1D Constant Advection coefficient. It runs for M time steps 
		and u is L periodic, i.e. u[0] = u[N] '''

	def Fourier_Transform(self,signal):
		''' Performs the fast fourier transform of a signal '''
		return np.fft.fft(signal)
	
	def Inverse_Fourier_Transform(self,fourier_transform):
		''' Performs the inverse fast fourier transform of a fourier transform '''
		return np.fft.ifft(fourier_transform)
	
	def run(self):
		
		# Grid:
		x_grid = self.grid_points(0.0,self.L,self.h)

		# Initial Condition:
		u_new = self.init_cond(x_grid)
		
		# Txt to store values
		a_file = open(self.File_Name+".txt", "w")
		np.savetxt(a_file,u_new)
		
		# Define the vector of fourier space 
		# Method 1:
		k_vec_space = np.fft.fftshift(np.arange(- self.N/2 , self.N/2 , 1))
		
		# Method 2:
		# k_vec_space = np.fft.fftfreq(self.N ,d = 1/self.N)

		# Multiplicative exponential array
		exponential_factor = np.exp((-2j * np.pi/self.L) * self.a * self.dt * k_vec_space)
		
		# Iteration over the time step sizes
		for k in range(1,self.M+1):
			
			u_old = u_new.copy()[:-1] # Change the u_new array to u_old
			u_old_FFT = self.Fourier_Transform(u_old) # Compute the FT
			
			u_new_FFT = u_old_FFT*exponential_factor # Evolve the function one step in the Fourier Space
			u_new = self.Inverse_Fourier_Transform(u_new_FFT).real # Compute the inverse FT
			
			u_new = np.append(u_new , u_new[0])
			np.savetxt(a_file, u_new) # Save the solution at time step k
			
		a_file.close()

class Initialize_Simulation:
	def __init__(self, initial_condition, x_min, x_max, N_x, v_min, v_max, N_v, T, M):

		# Initial Distribution Function: It can be a (N_v, N_x) array or (N_x, N_v) 
		self.initial_condition = initial_condition

		# X Dimension
		self.x_min = x_min # Left side of the container
		self.x_max = x_max # Right side of the container
		self.N_x = N_x # Divisions in the x dimension in N_x slots
		self.dx = (self.x_max - self.x_min)/self.N_x # Space step size

		# V Dimension
		self.v_min = v_min # Minimum velocity of the system
		self.v_max = v_max # Maximum velocity of the system
		self.N_v = N_v # Divisions in the v dimension in N_v slots
		self.dv = (self.v_max - self.v_min)/self.N_v

		# Time
		self.T = T # Max Time
		self.M = M # Divisions of time in M slots
		self.dt = self.T/self.M # Time step size

	def grid_points(self,start_point, end_point, step_size):
		'''Construct the grid points'''
		return np.arange(start_point,end_point + step_size,step_size)

class Advection_Methods_2D(Initialize_Simulation):
	'''Solves a 2 dimensional advection equation and integrates it in a time interval (0 -> T) 
		Input: 
		- initial_condition: Initial conditions function (x,v) 2 dimensional
		- x_min, x_max, N_x: Minimum and maximum positions of the container, dimension in the x space with N_x slts
		- v_min, v_max, N_v: Minimum and maximum velocity of the system, divisions in the v dimension in N_v slots
		- T, M: Maximum time and number of steps (M slots)
		- advcoeff_X, advcoeff_V: Advection Coefficient Vectors in the X dimension and V dimension
		- split_method: Splitting method to use: 'Lax Wendroff', 'Spectral' or 'Euler Upwind'
		- split_order: order of the splitting method: 1 or 2
		- file_name: file name to store the solutions '''

	def __init__(self, initial_condition, x_min, x_max, N_x, v_min, v_max, N_v, T, M, advcoeff_X, advcoeff_V, 
				split_method, split_order ,file_name):

		Initialize_Simulation.__init__(self, initial_condition, x_min, x_max, N_x, v_min, v_max, N_v, T, M)

		# Advection Coefficient Vectors
		self.advcoeff_X = advcoeff_X # advection coefficient vector in the X dimension 
		self.advcoeff_V = advcoeff_V # advection coefficient vector in the Y dimension

		# Method to use in splitting method
		self.split_method = split_method 

		# Order of the splitting method
		self.split_order = split_order
		
		# File Name Where solutions will be stored
		self.file_name = file_name
		
		# Total Mass Array: Array where masses will be stored
		self.TotalMass = []
		
		# L^2 Norm array: L2 norm will be stored
		self.L2_norm = []

	def Fourier_Transform(self,signal):
		''' Performs the fast fourier transform of a signal '''
		return np.fft.fft(signal,axis = 0) # indicates to transform by rows
	
	def Inverse_Fourier_Transform(self,fourier_transform):
		''' Performs the inverse fast fourier transform of a fourier transform '''
		return np.fft.ifft(fourier_transform,axis = 0) # indicates to transform by rows
	
	def Spectral_Method_2D(self , dtime , dist_func , advcoeff , s_min, s_max , N_size):
		''' Solves the advection equation with Spectral Scheme for one time step with vectorial 
			advection coefficient'''

		### Define the vector of fourier space ###
		
		# Method 1:
		k_vec_space = np.fft.fftshift(np.arange( -N_size/2 , N_size/2 , 1))
		# Method 2:
		# k_vec_space = np.fft.fftfreq(N_size, d = 1/N_size)
		
		k_vec_space = np.array([k_vec_space]).T # Change shape of k_vector
		
		# Multiplicative exponential array
		L_size = s_max - s_min
		exponential_factor = np.exp((-2j * np.pi/L_size) * dtime * k_vec_space * advcoeff)

		f_old = dist_func[:-1].copy() # Change the dist_func array to f_old and slice the array to N-1 spaces, consider periodic conditions
		f_old_FFT = self.Fourier_Transform(f_old) # Compute the FT
		
		f_new_FFT = f_old_FFT*exponential_factor # Evolve the function one step with the exponential factor        
		f_new = self.Inverse_Fourier_Transform(f_new_FFT).real # Compute the inverse FT

		f_new = np.append(f_new, [f_new[0]], axis = 0) # Append first row for periodic conditions: f[0] = f[N]

		return f_new

	def Lax_Wendroff_2D(self , dspace, dtime, dist_func, advcoeff):
		''' Solves the advection equation with Lax Wendroff Scheme for one time step with vectorial 
			advection coefficient'''

		# Multiplicative constant
		s_vector = (dtime/dspace)*advcoeff
		
		# Courant–Friedrichs–Lewy numerical stability condition for a vector
		if abs(s_vector).max() <= 1:
			
			# Advances Advection Equation in 1 time step
			f_old = dist_func[:-1].copy() # Slice the array to N-1 spaces, consider periodic conditions
		
			f_forward = np.roll(f_old,-1,axis=0) # u_j+1 (Vectorial Displacement to the left) 
			f_backward = np.roll(f_old,1,axis=0) # u_j-1 (Vectorial Displacement to the right)

			# Numerical scheme equation in two lines
			term = f_old - (s_vector/2.0) * (f_forward - f_backward)
			f_new =  term + (s_vector**2/2.0) * (f_forward - 2.0*f_old + f_backward)
			
			f_new = np.append(f_new, [f_new[0]], axis = 0) # Append first row for periodic conditions: f[0] = f[N]

			return f_new
		
		else: 
			raise ValueError('The Courant–Friedrichs–Lewy condition must be satisfied, a*dt/dx must be <= 1')

	def Euler_Upwind_2D(self, dspace, dtime, dist_func, advcoeff):
		''' Solves the advection equation with Euler Upwind Scheme for one time step with vectorial 
			advection coefficient'''

		s_vector = (dtime/dspace)*advcoeff # Define the CFL parameter vector
		
		# Courant–Friedrichs–Lewy numerical stability condition for a vector
		if abs(s_vector).max() <= 1:
			# Advances Advection Equation in 1 time step
			f_old = dist_func[:-1].copy() # define old distribution
			
			aux_sol = [] # define an auxiliary list to store solutions
			for idx,acoeff in enumerate(advcoeff):
				old_vector = np.array([f_old[:,idx]]).T # Extract the ith column of the distribution function
				
				# Conditions for periodic boundary conditions
				# Condition 1: a<0
				if acoeff < 0.0:
					v_forward = np.roll(old_vector, -1 , axis = 0) # u_j+1 (Vectorial Displacement to the left) 
					v_new = old_vector - s_vector[idx]*(v_forward - old_vector) # Formula for u(t+dt)
					aux_sol.append(v_new.T[0]) # Transpose the vector to row form
				
				# Condition 2: a>0
				elif acoeff > 0.0:
					v_backward = np.roll(old_vector, 1 ,axis = 0)  # u_j-1 (Vectorial Displacement to the right)
					v_new = old_vector - s_vector[idx]*(old_vector - v_backward) # Formula for u(t+dt)
					aux_sol.append(v_new.T[0]) # Transpose the vector to row form
					
			f_new = np.array(aux_sol).T
			f_new = np.append(f_new, [f_new[0]], axis = 0) # Append first row for periodic conditions: f[0] = f[N]

			return f_new # Return solution to original shape

		else:
			raise ValueError('The Courant–Friedrichs–Lewy condition must be satisfied, a*dt/dx must be <= 1')

	def integrate_distribution_function(self,distribution_func, shape_tuple, dimension_vec, dimension_string):
		''' This function integrates the distribution function f(x,v,L) with respect to one dimension.
			The input is a 2D array f(X,V) and it uses the trapezoidal method.
			distribution_func: Distribution function f(X,V), it can be with shape (X,V) or (V,X)
			shape_tuple: Shape of the distribution function, format: ('X','V') or ('V','X')
			dimension_vec: Dimension to integrate.
			dimension_string: Dimension string specification, 'X' or 'V'  '''
		
		if shape_tuple[0] == dimension_string:
			sol = integrate.trapz(y = distribution_func, x = dimension_vec , axis = 0)
			
		elif shape_tuple[1] == dimension_string:
			sol = integrate.trapz(y = distribution_func , x = dimension_vec , axis = 1)
	
		return np.array(sol)
	
	def integrate_full_distribution_func(self, func_array , shape_tuple , x_array , y_array ):
		''' This function integrates a function with shape (X,V) or (V,X) with respect to the two dimensions
			Inputs are a 2D array and the shape specification
			func_array: 2D Array or function to integrate
			shape_tuple: Shape of the array ('X','V') or ('V','X')
			x_array: Array of positions
			y_array: Array of velocities '''
		
		# First integrate over the X dimension:
		int1 = self.integrate_distribution_function(func_array, shape_tuple , x_array , 'X')
		# Second integrate over the V dimension: 
		int2 = integrate.trapz(y = int1, x = y_array )
		
		return int2

	def Spectral_Scheme_Splitting_O1(self,DistFunc):
		'''Solves the splitting method of order 1 with the Spectral Scheme for one time step (t -> t+dt)
		DistFunc: Dist function (shape: (Vdim,Xdim)) at time t'''
		
		# Solve Advection Equation in full time step for df/dt + advcoeff_X*df/dv with Spectral Scheme
		Adv_Spec_sol1 = self.Spectral_Method_2D(self.dt, DistFunc.copy(), self.advcoeff_X, self.v_min, self.v_max, self.N_v)
		Transposed_sol = Adv_Spec_sol1.T # Transpose the shape of the matrix
		# Solve Advection Equation in full time step for df/dt + advcoeff_V*df/dx with Spectral Scheme
		Adv_Spec_sol2 = self.Spectral_Method_2D(self.dt, Transposed_sol, self.advcoeff_V, self.x_min, self.x_max, self.N_x)
		new_distribution = Adv_Spec_sol2.T # Get back the original shape of the distribution function

		return new_distribution

	def Lax_Wendroff_Splitting_O1(self, DistFunc):
		'''Solves the splitting method of order 1 with the Lax wendroff Scheme for one time step (t -> t+dt)
		DistFunc: Dist function (shape: (Vdim,Xdim)) at time t '''

		# Solve Advection Equation in full time step for df/dt + advcoeff_X*df/dv with Lax Wendroff Scheme
		Adv_Lax_sol1 = self.Lax_Wendroff_2D(self.dv, self.dt , DistFunc.copy(), self.advcoeff_X)
		Transposed_sol = Adv_Lax_sol1.T # Transpose the shape of the matrix to (Xdim,Vdim)
		# Solve Advection Equation in full time step for df/dt + advcoeff_V*df/dx with Lax Wendroff
		Adv_Lax_sol2 = self.Lax_Wendroff_2D(self.dx, self.dt, Transposed_sol, self.advcoeff_V)
		new_distribution = Adv_Lax_sol2.T # Get back the original shape of the distribution function
		
		return new_distribution
	
	def Euler_Upwind_Splitting_O1(self, DistFunc):
		'''Solves the splitting method of order 1 with the Euler Upwind Scheme for one time step (t -> t+dt)
		DistFunc: Dist function (shape: (Vdim,Xdim)) at time t '''

		# Solve Advection Equation in half time step for df/dt - E*df/dv with Euler Upwind Scheme
		Adv_Euler_sol1 = self.Euler_Upwind_2D(self.dv, self.dt, DistFunc.copy(), self.advcoeff_X)
		Transposed_sol = Adv_Euler_sol1.T # Transpose the shape of the matrix to (Xdim,Vdim)
		# Solve Advection Equation in half time step for df/dt + v*df/dx with Euler Upwind Scheme
		Adv_Euler_sol2 = self.Euler_Upwind_2D(self.dx, self.dt, Transposed_sol, self.advcoeff_V)
		new_distribution = Adv_Euler_sol2.T # Get back the original shape of the distribution function
		
		return new_distribution
		
	def Splitting_Method_Order1(self,Method_Str,DistFunc):
		'''Implements the Splitting method of order 1 with 3 posible differential schemes:
		Method_Str: String specifying the method to implement, it can be: Lax Wendroff, Spectral or Euler Upwind 
		DistFunc: Distribution function (shape: (Vdim,Xdim)) at time t'''
		
		if Method_Str == 'Spectral':
			return self.Spectral_Scheme_Splitting_O1(DistFunc)
		elif Method_Str == 'Lax Wendroff':
			return self.Lax_Wendroff_Splitting_O1(DistFunc)
		elif Method_Str == 'Euler Upwind':
			return self.Euler_Upwind_Splitting_O1(DistFunc)
	
	def Spectral_Scheme_Splitting_O2(self,DistFunc):
		'''Solves the splitting method of order 2 (Strang splitting method) with the Spectral Scheme for
		   one time step (t -> t+dt)
		DistFunc: Dist function (shape: (Vdim,Xdim)) at time t '''

		# Solve Advection Equation in half time step for df/dt + advcoeff_X*df/dv with Spectral Scheme
		Adv_Spec_sol1 = self.Spectral_Method_2D(self.dt/2, DistFunc.copy(), self.advcoeff_X, self.v_min, self.v_max, self.N_v)
		Transposed_sol1 = Adv_Spec_sol1.T # Transpose the shape of the matrix
		# Solve Advection Equation in full time step for df/dt + advcoeff_V*df/dx with Spectral Scheme
		Adv_Spec_sol2 = self.Spectral_Method_2D(self.dt, Transposed_sol1, self.advcoeff_V, self.x_min, self.x_max, self.N_x)
		Transposed_sol2 = Adv_Spec_sol2.T # Get back the original shape of the distribution function	
		# Solve Advection Equation in half time step for df/dt - E*df/dv with Spectral Scheme
		new_distribution = self.Spectral_Method_2D(self.dt/2 , Transposed_sol2 , self.advcoeff_X, self.v_min, self.v_max, self.N_v)

		return new_distribution	

	def Lax_Wendroff_Splitting_O2(self,DistFunc):
		'''Solves the splitting method of order 2 (Strang splitting method) with the Lax wendroff Scheme 
			for one time step (t -> t+dt).
		DistFunc: Dist function (shape: (Vdim,Xdim)) at time t '''

		# Solve Advection Equation in half time step for df/dt - advcoeff_X*df/dv with Lax Wendroff
		Adv_Lax_sol1 = self.Lax_Wendroff_2D(self.dv , self.dt/2 , DistFunc.copy() , self.advcoeff_X)
		Transposed_sol1 = Adv_Lax_sol1.T # Transpose the shape of the matrix to (Xdim,Vdim)
		# Solve Advection Equation in full time step for df/dt + advcoeff_V*df/dx with Lax Wendroff
		Adv_Lax_sol2 = self.Lax_Wendroff_2D(self.dx , self.dt , Transposed_sol1 , self.advcoeff_V)
		Transposed_sol2 = Adv_Lax_sol2.T # Get back the original shape of the distribution function
		# Solve Advection Equation in half time step for df/dt - advcoeff_X*df/dv with Lax Wendroff
		new_distribution = self.Lax_Wendroff_2D(self.dv , self.dt/2 , Transposed_sol2 , self.advcoeff_X)
	
		return new_distribution
	
	def Euler_Upwind_Splitting_O2(self,DistFunc):
		'''Solves the splitting method of order 2 (Strang splitting method) with the Euler Upwind Scheme
			for one time step (t -> t+dt)
		DistFunc: Dist function (shape: (Vdim,Xdim)) at time t '''

		# Solve Advection Equation in half time step for df/dt - advcoeff_X*df/dv with Euler Upwind Scheme
		Adv_Euler_sol1 =  self.Euler_Upwind_2D(self.dv, self.dt/2, DistFunc.copy(), self.advcoeff_X)
		Transposed_sol1 = Adv_Euler_sol1.T # Transpose the shape of the matrix to (Xdim,Vdim)
		# Solve Advection Equation in full time step for df/dt + advcoeff_V*df/dx with Euler Upwind Scheme
		Adv_Euler_sol2 =  self.Euler_Upwind_2D(self.dx, self.dt, Transposed_sol1, self.advcoeff_V)
		Transposed_sol2 = Adv_Euler_sol2.T # Get back the original shape of the distribution function
		# Solve Advection Equation in half time step for df/dt - advcoeff_X*df/dv with Euler Upwind Scheme
		new_distribution =  self.Euler_Upwind_2D(self.dv, self.dt/2, Transposed_sol2, self.advcoeff_X)

		return new_distribution
	
	def Splitting_Method_Order2(self,Method_Str,DistFunc):
		'''Implements the Strang Splitting Method (order 2) with 3 posible differential schemes:
		Method_Str: String specifying the method to implement, it can be: Lax Wendroff, Spectral or Euler Upwind 
		DistFunc: Distribution function (shape: (Vdim,Xdim)) at time t '''
		
		if Method_Str == 'Spectral':
			return self.Spectral_Scheme_Splitting_O2(DistFunc)
		elif Method_Str == 'Lax Wendroff':
			return self.Lax_Wendroff_Splitting_O2(DistFunc)
		elif Method_Str == 'Euler Upwind':
			return self.Euler_Upwind_Splitting_O2(DistFunc)
		
	def Splitting_Order(self, Order, Method_Str, DistFunc):
		if Order == 1:
			return self.Splitting_Method_Order1(Method_Str,DistFunc)
		elif Order == 2:
			return self.Splitting_Method_Order2(Method_Str,DistFunc)

	def run_iteration(self):
		'''Run the simulations for T steps'''
		
		# Position and Velocity vectors
		positions = self.grid_points(self.x_min, self.x_max, self.dx)
		velocities = self.grid_points(self.v_min, self.v_max, self.dv)
		
		# Meshgrid of the coordinates in phase space
		Pos, Vel = np.meshgrid(positions, velocities)
		
		# Initial Distribution Function
		new_dist = self.initial_condition(Pos,Vel) # Function with dim (Vel dim, Pos dim)

		# Txt to store values
		a_file = open(self.file_name+".txt", "w")
		np.savetxt(a_file,new_dist)

		for k in range(self.M):

			old_dist = new_dist.copy() # Replace the new solution by the old
			
			# Calculate the total mass
			total_mass = self.integrate_full_distribution_func(old_dist,('V','X'),positions,velocities)
			self.TotalMass.append(total_mass)
			
			# Calculate the L2 norm
			l2_norm = self.integrate_full_distribution_func(old_dist**2,('V','X'),positions,velocities)
			self.L2_norm.append(l2_norm)
			
			### Splitting Algorithm ###
			new_dist = self.Splitting_Order(self.split_order, self.split_method, old_dist)

			np.savetxt(a_file, new_dist) # Save the solution at time step k

		a_file.close()
		
		return self.TotalMass, self.L2_norm