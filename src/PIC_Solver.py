import numpy as np
import PoissonSolvers
from scipy.stats import uniform,norm

class Initialize_Particles:
	''' Initialize distribution of particles 
		- Nk: Number of markers or particles in the simulation
		- x_min, x_max: Minimum and maximum positions of the container
		- v_min, v_max, N_v: Minimum and maximum velocity of the system
		- x_distribution, v_distribution: Initial position and velocity sampling of particles, it can be uniform or normal
		- x_distribution_params, y_distribution_params: An array with 2 parameters:
								If x,y_distribution = 'uniform': x,y_distribution_params = [x,v_min; x,v_max - x,v_min = length]
								If x,y_distribution = 'normal': x,y_distribution_params = [mean, standard_deviation]

		- charge, mass: particles adimensional charge and mass. For electrons (charge = -1, mass = 1)'''

	def __init__(self, Nk, x_min, x_max, x_distribution, x_distribution_params, v_min, v_max, v_distribution, 
				v_distribution_params, charge, mass):
		
		self.Nk = Nk # Number of particles or markers in the simulation
		self.x_max = x_max
		self.x_min = x_min
		self.v_max = v_max
		self.v_min = v_min
		self.charge = charge 
		self.mass = mass

		# Parameters of the distributions:
		self.x_distribution_params = x_distribution_params
		self.v_distribution_params = v_distribution_params

		if x_distribution == 'uniform':
			# The positions distribution comes from a uniform distribution with a = 0 and b = L_x
			self.positions_sampling = uniform.rvs(size = self.Nk, loc = self.x_distribution_params[0], scale = self.x_distribution_params[1])
		elif x_distribution == 'normal':
			pos_sampling = norm.rvs(size = self.Nk, loc = self.x_distribution_params[0], scale = self.x_distribution_params[1])
			self.positions_sampling = (pos_sampling - self.x_min)%(self.x_max - self.x_min) + self.x_min  # Periodic Condition
		
		if v_distribution == 'normal':
			# The velocities distribution comes from a normal distribution with mu = 0 and sigma = 1.
			vel_sampling = norm.rvs(size = self.Nk, loc = self.v_distribution_params[0], scale = self.v_distribution_params[1])
			self.velocities_sampling = (vel_sampling - self.v_min)%(self.v_max - self.v_min) + self.v_min  # Periodic Condition
		elif v_distribution == 'uniform':
			self.velocities_sampling = uniform.rvs(size = self.Nk, loc = self.v_distribution_params[0], scale = self.v_distribution_params[1])

class Initialize_Grid:
	''' Initialize the grid where the fields and densities will be calculated 
		- x_min, x_max, N_x: Minimum and maximum positions of the container, dimension in the x space with N_x slts
		- v_min, v_max, N_v: Minimum and maximum velocity of the system, divisions in the v dimension in N_v slots '''		
	def __init__(self, x_min, x_max, N_x, v_min, v_max, N_v):
		
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

	def grid_points(self,start_point, end_point, step_size):
		'''Construct the grid points'''
		return np.arange(start_point,end_point + step_size,step_size)


class PIC(Initialize_Particles, Initialize_Grid):
	''' Class that implements a solver for an Electrostatic Particle in Cell algorithm. 
		Inputs:
		- T, M: Maximum time and number of steps (M slots)
		- sampling_distribution: a function that describes the initial distribution of positions and velocities of al particles
		- control_variate: function that controls the variation of the density of particles
		- spline_degree: degree of the B-spline interpolation to calculate the macroparticles properties in the grid '''
	def __init__(self, N_k, charge, mass, x_min, x_max, N_x, x_distribution, x_distribution_params, v_min, v_max, N_v, 
				v_distribution, v_distribution_params, T, M, sampling_distribution, control_variate, spline_degree, file_name):

		Initialize_Particles.__init__(self, N_k, x_min, x_max, x_distribution, x_distribution_params, v_min, v_max, 
										v_distribution, v_distribution_params, charge, mass)
		Initialize_Grid.__init__(self, x_min, x_max, N_x, v_min, v_max, N_v)

		# Time
		self.T = T # Max Time
		self.M = M # Divisions of time in M slots
		self.dt = self.T/self.M # Time step size

		# Sampling distribution: function of x and v
		self.sampling_distribution = sampling_distribution

		# Control Variate function
		self.control_variate = control_variate

		# B spline degree and index
		self.spline_degree = spline_degree # degree of spline interpolation of macro particles, can be 1,2,3,...
		self.spline_index = 0 # index of spline, can take only value 0

		self.file_name = file_name

	# Function to define knot vector generation function 
	def knot_function(self, k, deltax):
		'''Knot vector generation function. 
			- k: B-spline degree
			- deltax: knot spacing '''

		return deltax * np.arange(-(k + 1)/2, (k + 1)/2 + 1, 1)

	def bspline(self, x, k, j, knots):
		'''Definition of the Bspline basis function B_j^k vectorized with a given knot array
		Inputs:
			x: vector in which the basis function will be evaluated
			k: B-spline degree
			j: B-spline index
			knots: knot points or vector'''

		# This algorithm executes the recursive definition of b spline with a vector modification

		if k == 0:
			return np.where(np.logical_and(x >= knots[j], x < knots[j+1]), np.ones(x.shape), np.zeros(x.shape))
		
		elif k > 0:
			
			if knots[j+k] == knots[j]:
				c1 = np.zeros(x.shape)
			
			else:
				c1 = (x - knots[j])/(knots[j+k] - knots[j]) * self.bspline(x,k-1,j,knots)
			
			if knots[j+k+1] == knots[j+1]:
				c2 = np.zeros(x.shape)
			
			else:
				c2 = (knots[j+k+1] - x)/(knots[j+k+1] - knots[j+1]) * self.bspline(x,k-1,j+1,knots)
				
			return c1 + c2

	def find_indexes(self, grid, coordinates):
		''' Function to find the indexes of each particle coordinates x or v in the grid mesh
		grid: grid mesh, it can be the positions grid or velocities grid
		coordinates: the particles positions coordinates or velocities coordinates'''

		Grid,Coordinates =  np.meshgrid(grid, coordinates) # Create mesh grid to calculate the distances for each coordinate
														   # with respect to the grid
		distances = abs(Coordinates - Grid) # calculate the absolute value of the distances
		sorted_dists_idx = np.argsort(distances) # we sort the distances from minimum to max and take the indexes of that
												 # sorting process
		min_vals = sorted_dists_idx[:,:2] # Take the two first minimum values
		# Return the array of  minimum values between the min vals
		return np.amin(min_vals,axis=1)

	def calculate_weights(self, x_grid, v_grid, particle_positions, particle_velocities, hist2D):
		'''Function to calculate the weights of all particles:
		x_grid: positions grid
		v_grid: velocities grid
		particle_positions: positions of all particles
		particle_velocities: velocities of all particles
		hist2D: Two dimensional histogram of all particles in the grid (shape = (V,X))'''

		idx_ys = self.find_indexes(v_grid,particle_velocities) # velocities indexes
		idx_xs = self.find_indexes(x_grid,particle_positions) # positions indexes

		# Evaluate the indices in the 2D particle histogram
		return hist2D[idx_ys,idx_xs]

	def particle_density(self, x_t, v_t, x_0, v_0, x_mesh, Lx, weights, Nk, bs_func, k_vec, sp_idx, spl_degree, control_var, samp_dist):
		''' This function calculates the particle density in each mesh point due to the contribution of all particles in 
			the simulation at time t.
			x_t: Particle positions at time t
			v_t: Particle velocities at time t
			x_0: Initial positions
			v_0: Initial velocities
			x_mesh: x grid mesh
			Lx: Positions size box u[0] = u[Lx] 
			weights: particle weights
			Nk: Number of markers (number of particles in simulation)
			bs_func: bspline function
			k_vec: knots vector
			sp_idx: spline element index
			spl_degree: spline degree '''

		# The density at the position x_j with index j in the mesh is calculated by the formula:
		#  N_j(t) = 1 + 1/Nk* Sum_{k = 1}^{Nk}{Spline(x_j - x_k) * [wk - control_var(v_k)/samp(x_0,v_0)]}  

		# Verify if the dimensions match
		if x_t.shape[0] != Nk and v_t.shape != Nk:
			raise ValueError("Positions and Velocities must have the dimension of markers")
		if x_0.shape[0] != Nk and v_0.shape != Nk:
			raise ValueError("Positions and Velocities must have the dimension of markers")
		if weights.shape[0] != Nk:
			raise ValueError("Weights must have the dimension of markers")
		
		# Create a mesh grid of the particle positions and the grid points
		X_mesh, X_t = np.meshgrid(x_mesh,x_t) 
		
		# Calculate the spline interpolation of each particle in each point in the grid (matrix form)
		spline_factor = bs_func((x_mesh - X_t)%Lx, spl_degree, sp_idx, k_vec) # Shape (Nk,Nx)

		# The second factor takes into account the weights of each particle, the control variate and the sampling distribution
		second_factor = weights - control_var(v_t)/samp_dist(x_0,v_0) # Shape (Nk,)
		
		# Sumation of all terms and determination of the density at the grid points
		term1 = 1.0
		term2 = (spline_factor.T * second_factor).T # change the shape of the spline matrix to match the second factor dimension
		term2 = (1.0/Nk)*sum(term2) # sum over all the particles contributions to the density in the grid
		
		return term1 + term2

	def electric_field_particle_points(self, x_t, Nk, x_mesh, efield, Nx, Lx, bs_func, k_vec, sp_idx, spl_degree):
		''' This function calculates the electric field in the particle positions with the contribution of 
			the electric field in each mesh point in the simulation at time t.
			x_t: Particle positions at time t
			x_mesh: x grid mesh
			efield: electric field evaluated in the mesh points at time t
			Lx: Positions size box u[0] = u[Lx]
			Nk: Number of markers (number of particles in simulation)
			bs_func: bspline function
			k_vec: knots vector
			sp_idx: spline element index
			spl_degree: spline degree '''

		# The electric field at the particle position x_k using the contribution of the electric field at all grid
		# points is calculated with the formula:
		#  E_h(t,x_k) = dx * Sum_{j = 1}^{Nk}{efield(t, x_j) * Spline(x_k - x_j)}

		if x_t.shape[0] != Nk:
			raise ValueError("Positions must have the dimension of markers")
		if x_mesh.shape[0] - 1 != Nx:
			raise ValueError("Mesh grid doesnt match the Nx slots")
		if efield.shape[0] - 1 != Nx:
			raise ValueError("Electric field dimension doesnt match Nx slots")
		
		deltax = Lx/Nx # Define the dx value
		
		X_t, X_mesh = np.meshgrid(x_t, x_mesh) # mesh grid for the particle positions and the mesh coordinates
		_, Efield = np.meshgrid(x_t, efield) # mesh grid for the particle positions and the electric field
		
		spline_factor = bs_func((X_t - X_mesh)%Lx, spl_degree, sp_idx, k_vec) # Shape (Nx,Nk)
		second_factor = spline_factor * Efield # Multiply the spline matrix by the electric field. Shape (Nx, Nk)
		term = sum(second_factor) # We sum over all grid points 
		term = term * deltax # multiply by dx

		return term

	def velocity_advance(self, velocity, electric_field, dt, q, m):
		''' Advances the velocity in one time step
			velocity: velocities of all particles at any time
			electric_field: electric field evaluated at all particles at any time
			dt: time step
			q: particle charge (electrons: q = -1) adimensional
			m: particle mass (electrons: m = 1) adimensional '''
		velocity_forward = (velocity + (q*dt/(2*m)) * electric_field)  
		velocity_forward = (velocity_forward - self.v_min)%(self.v_max - self.v_min) + self.v_min  # Periodic Condition

		return velocity_forward

	def position_advance(self, position, velocity, dt):
		''' Advances the position in one time step
			position: positions of all particles at any time
			velocity: velocities of all particles at any time
			dt: time step'''
		position_forward = position + dt*velocity
		position_forward = (position_forward - self.x_min)%(self.x_max - self.x_min) + self.x_min  # Periodic Condition
		
		return position_forward

	def electric_field_advance(self, part_x_forward, part_v_half, x_0, v_0, x_grid, xmin, xmax, Nx, v_grid, N_k, 
								bsfunc, knot_vec, spl_idx, spl_degree, control_var, samp_dist):
		
		''' Advance the electric field of evaluated in the particle positions one time step
		part_x_forward: positions of the particles at time t_n+1
		part_v_half: velocities of the particles at time t_n+1/2
		x_0: initial particle positions
		v_0: initial particle velocities
		x_grid, v_grid: x and v mesh grids respectively
		xmin, xmax: min and max positions of the grid
		Nx: number of slots in the positions grid
		N_k: number of particles in simulation
		bsfunc: bspline interpolation function
		knot_vec: knot vector 
		spl_idx: spline index
		spl_degree: spline degree
		control_var: variable control function
		samp_dist: sampling distribution of functions '''

		### Histogram of particles evolved in one time step
		# Define the histogram of the particles distribution in phase space
		hist2D, xedges, yedges = np.histogram2d(part_x_forward, part_v_half, 
													bins = (x_grid,v_grid), density = True)
		hist2D = hist2D.T # transpose to have a shape of (N_v, N_x)
		
		# Calculate the particle weights
		wk = self.calculate_weights(x_grid, v_grid, part_x_forward, part_v_half, hist2D)

		# Calculate the density at half time step
		density = self.particle_density(part_x_forward, part_v_half, x_0, v_0, x_grid, xmax-xmin, wk,
										N_k, bsfunc, knot_vec, spl_idx, spl_degree, control_var, samp_dist)

		# Rho vector for Poisson Equation
		rho = 1.0 - density
		
		# Solve Poisson Equation and determine the electric potential
		phi_solution = PoissonSolvers.Poisson1D_Periodic_BC_Solver1(xmin, xmax, Nx, rho).run()
		
		# Calculate the initial Electric Field at grid points
		e_field = -np.gradient(phi_solution, x_grid)

		# Calculate the initial Electric Field at the particle positions
		efield_particles = self.electric_field_particle_points(part_x_forward, N_k, 
									x_grid, e_field, Nx, xmax - xmin, bsfunc, knot_vec, spl_idx, spl_degree)

		return efield_particles

	def verlet_pusher(self, x_n, v_n, E_n, dt, q, m, x_0, v_0, x_grid, xmin, xmax, Nx, v_grid, N_k, bsfunc, knot_vec,
		spl_idx, spl_degree, control_var, samp_dist):

		''' Verlet pusher: advance the particles position and velocity in one time step
		x_n: positions of all particles at time t = t_n
		v_n: velocities of all particles at time t = t_n
		E_n: Electric field in all particles at time t = t_n
		dt: time step
		q: particle charge (electrons: q = -1) adimensional
		m: particle mass (electrons: m = 1) adimensional'''
		
		# The verlet pusher advances the particles in two time steps
		# In the first step we calculate the positions at time t_n+1 and velocity at time t_n+1/2
		v_half = self.velocity_advance(v_n, E_n, dt, q, m) # velocity at half time step
		x_forward = self.position_advance(x_n, v_half, dt) # position at time t_n+1

		# In the second step we evolve the electric field at half time step
		E_forward = self.electric_field_advance(x_forward, v_half, x_0, v_0, x_grid, xmin, xmax, Nx, v_grid, 
			N_k, bsfunc, knot_vec, spl_idx, spl_degree, control_var, samp_dist)
		
		# In the third step we calculate the velocity at time t_n+1
		v_forward = self.velocity_advance(v_half, E_forward, dt, q, m)
		
		return x_forward, v_forward, E_forward

	def run_iteration(self):

		# Position and Velocity vectors at time t = 0
		positions_grid = self.grid_points(self.x_min, self.x_max, self.dx)
		velocities_grid = self.grid_points(self.v_min, self.v_max, self.dv)

		# Initial particle coordinates in phase space
		coordinates0 = np.stack((self.positions_sampling, self.velocities_sampling))

		# Define the knots of the bspline function
		knots = self.knot_function(self.spline_degree, self.dx) 

		#### We Calculate the Electric Field at time 0 defined at all particle positions ####

		electric_field_new = self.electric_field_advance(self.positions_sampling, self.velocities_sampling, 
			self.positions_sampling, self.velocities_sampling, positions_grid, self.x_min, self.x_max, 
			self.N_x, velocities_grid, self.Nk , self.bspline, knots, self.spline_index, self.spline_degree, 
			self.control_variate, self.sampling_distribution)

		# Txt to store values
		a_file = open(self.file_name+".txt", "w")
		np.savetxt(a_file, coordinates0)

		# Create actualization variables
		particle_position_new = self.positions_sampling
		particle_velocity_new = self.velocities_sampling

		# Begin Loop:
		for k in range(self.M):

			particle_position_old = particle_position_new.copy()
			particle_velocity_old = particle_velocity_new.copy()
			electric_field_old = electric_field_new.copy()

			# Verlet Pusher to advance particles in one time step:
			particle_position_new, particle_velocity_new, electric_field_new = self.verlet_pusher(particle_position_old, 
				particle_velocity_old, electric_field_old, self.dt, self.charge, self.mass, self.positions_sampling, 
				self.velocities_sampling, positions_grid, self.x_min, self.x_max, self.N_x, velocities_grid, self.Nk, 
				self.bspline, knots, self.spline_index, self.spline_degree, self.control_variate, self.sampling_distribution)

			coordinates = np.stack((particle_position_new, particle_velocity_new))
			np.savetxt(a_file, coordinates) # Save the particle coordinates at time step k

		a_file.close()