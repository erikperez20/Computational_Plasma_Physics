import numpy as np
from scipy import integrate
import PoissonSolvers
import AdvectionSolvers

class VlasovPoisson1D1V(AdvectionSolvers.Initialize_Simulation):

	def __init__(self, initial_condition, x_min, x_max, N_x, v_min, v_max, N_v, T, M, split_method, split_order ,file_name):

		AdvectionSolvers.Initialize_Simulation.__init__(self, initial_condition, x_min, x_max, N_x, v_min, v_max, N_v, T, M)

		# Method to use in splitting method
		self.split_method = split_method 

		# Order of the splitting method
		self.split_order = split_order
		
		# File Name Where solutions will be stored
		self.file_name = file_name
		
		# Empty lists to store parameters
		self.TotalMass = [] # Total Mass Array: Array where masses will be stored
		self.TotalMomentum = [] # Total Momentum Array: Array where total momentum will be stored
		self.TotalEnergy = [] # Total Energy Array: Array where the total energy will be stored
		self.L2_norm = [] # L^2 Norm array
		self.EField = [] # Electric Field

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

	def run_iteration(self):

		# Position and Velocity vectors
		positions = self.grid_points(self.x_min, self.x_max, self.dx)
		velocities = self.grid_points(self.v_min, self.v_max, self.dv)

		Pos, Vel = np.meshgrid(positions, velocities) # Meshgrid of the coordinates in phase space

		# Initial Distribution Function
		new_dist = self.initial_condition(Pos, Vel) # Function with dim (Vel dim, Pos dim)

		# Txt to store values
		a_file = open(self.file_name+".txt", "w") # open file
		np.savetxt(a_file,new_dist)
			
		for k in range(self.M):
			
			old_dist = new_dist.copy() # Replace the new solution by the old
			
			#### Solve Poisson Equation ###
			# Integrate distribution function in velocity space
			intfunc = self.integrate_distribution_function(old_dist,('V','X'), velocities,'V')
			rho_den = 1.0 - intfunc # Define density array
			# Solve Poisson Equation with Periodic Boundary Conditions
			Poisson_sol =  PoissonSolvers.Poisson1D_Periodic_BC_Solver1(self.x_min, self.x_max, self.N_x, rho_den).run()
			
			#### Store all parameters ####
			# Solve electric field equation with the electric potential solution from Poisson Equation
			E_field = -np.gradient(Poisson_sol, positions)
			self.EField.append(E_field)
			# Calculate the total mass
			total_mass = self.integrate_full_distribution_func(old_dist, ('V','X'), positions, velocities)
			self.TotalMass.append(total_mass)
			# Calculate the total momentum
			f_velocity = (old_dist.T * velocities).T # Dist. f times the velocity vector
			total_momentum = self.integrate_full_distribution_func(f_velocity, ('V','X'), positions, velocities)
			self.TotalMomentum.append(total_momentum)
			# Calculate the total energy
			f_velocity_sqr = 0.5*(old_dist.T * (velocities**2)).T # Dist. f times the kinetic energy vector
			energy1 = self.integrate_full_distribution_func(f_velocity_sqr, ('V','X'), positions, velocities) # Contribution by f
			energy2 = self.integrate_distribution_function((E_field**2)*0.5, ('X'), positions, 'X')# Contribution of Efield
			total_energy = energy1 + energy2 # Sum of both energies
			self.TotalEnergy.append(total_energy)
			# Calculate the L2 norm
			l2_norm = self.integrate_full_distribution_func(old_dist**2, ('V','X'), positions, velocities)
			self.L2_norm.append(l2_norm)

			#### Solve the 2D Advection Equation #### 
			# Create the Advection Object
			advection_obj = AdvectionSolvers.Advection_Methods_2D(self.initial_condition, self.x_min, self.x_max, self.N_x, 
							self.v_min, self.v_max, self.N_v, self.T, self.M, -E_field, velocities, self.split_method, 
							self.split_order, self.file_name)
			new_dist = advection_obj.Splitting_Order(self.split_order, self.split_method, old_dist) # Splitting Algorithm 
			
			np.savetxt(a_file, new_dist) # Save the solution at time step k
		a_file.close()
		
		return self.TotalMass , self.TotalMomentum , self.TotalEnergy , self.L2_norm , self.EField

