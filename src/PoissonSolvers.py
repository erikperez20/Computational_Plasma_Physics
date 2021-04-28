import numpy as np
import matplotlib.pyplot as plt

# In this module 3 types of Poisson Solvers are developed: Dirichlet Boundary Conditions, Neumann Boundary Conditions,
# Mixed Boundary Conditions and Periodic Boundary Conditions. All are solved via the finite difference method to first order approximation.

class Poisson1D_Dirichlet_BC_Solver:

	''' Class that solves the Poisson Equation with Dirichlet BC. -phi''(x) = rho(x)
		Inputs:
			- a,b: Left and right extreme points of the grid 
			- alpha,beta: Boundary conditions at x = a and x = b
			- N: Number of mesh elements
			- rho: Density function '''

	def __init__(self, a , b , alpha , beta , N , rho):

		# Extreme points x0 = a, xN = b
		self.a = a
		self.b = b
		
		# Boundary conditions:
		self.alpha = alpha
		self.beta = beta
		
		# Number of mesh elements
		self.N = N
		
		# Density function
		self.rho = rho
		
		# Step size:
		self.h = self.b/self.N

	def grid_points(self,start_point, end_point, step_size):
		'''Construct the grid points'''
		return np.arange(start_point , end_point + step_size , step_size)
	
	def CalculateEigenSystem(self,A_matrix):
		EigenVals, EigenVec = np.linalg.eig(A_matrix)
		return EigenVals, EigenVec
	
	# We define a method to construct a tridiagonal matrix:
	def A_Matrix(self, val1, val2, val3, k1 = 0, k2 = 1, k3 = -1):
		# The method constructs three diagonal matrices with k_i indicating the shift from the main diagonal
		''' val1,2,3: 1-D arrays
			k1,2,3: Diagonals where the arrays will be positioned'''
		
		tridiagonal_mat = np.diag(val1, k1) + np.diag(val2, k2) + np.diag(val3, k3)
		A_mat = (1/self.h**2)*tridiagonal_mat
		return A_mat
	
	def R_Matrix(self,g_points):
		''' We construct the R matrix to solve the Ax=R system of equations. We define the function rho applied to the
		grid points and then replace the end points with the boundary conditions.'''

		matrix = self.rho(g_points[1:-1]) # New modification
		matrix[0] = matrix[0] + self.alpha/self.h**2
		matrix[self.N - 2] = matrix[self.N - 2] + self.beta/self.h**2
		return matrix
	
	def LinearSolver(self,A_matrix,R_matrix):
		''' We solve the linear system of equations A_matrix*X = R_matrix for X, then insert the boundary conditions
		at the start and end of the array'''
		
		sol = list(np.array(np.linalg.solve(A_matrix,R_matrix)))
		sol.append(self.beta)
		sol.insert(0,self.alpha)
		return sol
		
	def run(self):
		
		x_grid = self.grid_points(self.a , self.b , self.h)

		diagonal1 = 2*np.ones(self.N - 1 , dtype = float)
		diagonal2 = -1*np.ones(self.N - 2 , dtype = float)
		diagonal3 = -1*np.ones(self.N - 2 ,dtype = float)
		AMatrix = self.A_Matrix(diagonal1,diagonal2,diagonal3,0,1,-1)
		
		RMatrix = self.R_Matrix(x_grid)
		
		solution = self.LinearSolver(AMatrix,RMatrix)
		
		return solution


class Poisson1D_Mixed_BC_Solver1:

	''' Class that solves the Poisson Equation with Mixed BC. -phi''(x) = rho(x)
		Inputs:
			- a,b: Left and right extreme points of the grid 
			- alpha: Boundary condition at x = a for phi(a) = alpha
			- gamma: Boundary condition at x = b for phi'(b) = gamma
			- N: Number of mesh elements
			- rho: Density function  '''

	def __init__(self, a , b , alpha , gamma , N , rho):
		
		# Extreme points x0 = a, xN = b
		self.a = a
		self.b = b
		
		# Boundary conditions:
		self.alpha = alpha # phi(0) = alpha
		self.gamma = gamma # phi'(N) = gamma
		
		# Number of mesh elements
		self.N = N
		
		# Density function
		self.rho = rho
		
		# Step size:
		self.h = self.b/self.N

	def grid_points(self,start_point, end_point, step_size):
		'''Construct the grid points'''
		return np.arange(start_point , end_point + step_size , step_size)
	
	def CalculateEigenSystem(self,A_matrix):
		EigenVals, EigenVec = np.linalg.eig(A_matrix)
		return EigenVals, EigenVec
	
	# We define a method to construct a tridiagonal matrix:
	def A_Matrix(self,N_Dim ,val1, val2, val3, k1=0, k2=1, k3=-1):
		# The method constructs three diagonal matrices with k_i indicating the shift from the main diagonal
		''' val1,2,3: 1-D arrays
			k1,2,3: Diagonals where the arrays will be positioned'''
		
		tridiagonal_mat = np.diag(val1, k1) + np.diag(val2, k2) + np.diag(val3, k3) # Three main diagonals
		
		# Modify the last row of the matrix where the calculation of the derivative is done
		tridiagonal_mat[N_Dim-1][N_Dim-1] = 1 # Modify last element from last row and last column to 1
		tridiagonal_mat[N_Dim-1][N_Dim-2] = 0 
		tridiagonal_mat[N_Dim-1][N_Dim-3] = -1
		
		A_mat = (1/self.h**2)*tridiagonal_mat # Multiply by the factor of 1/h^2
		
		return A_mat
	
	def R_Matrix(self,g_points):
		''' We construct the R matrix to solve the Ax=R system of equations. We define the function rho applied to the
		grid points and then replace the end points with the boundary conditions.'''
		
		matrix = self.rho(g_points[1:])
		matrix[0] = matrix[0] + self.alpha/self.h**2
		matrix[self.N - 1] = 2.0*self.gamma/self.h
		return matrix
	
	def LinearSolver(self,A_matrix,R_matrix):
		''' We solve the linear system of equations A_matrix*X = R_matrix for X, then insert the boundary conditions
		at the start and end of the array'''
		
		sol = list(np.array(np.linalg.solve(A_matrix,R_matrix)))
		sol.insert(0,self.alpha)
		return sol
	
	def run(self):

		x_grid = self.grid_points(self.a , self.b , self.h)
		
		diagonal1 = 2*np.ones(self.N  , dtype = float) # Main diagonal must be of length N
		diagonal2 = -1*np.ones(self.N - 1 , dtype = float)
		diagonal3 = -1*np.ones(self.N - 1 ,dtype = float)
		AMatrix = self.A_Matrix(self.N , diagonal1 , diagonal2 , diagonal3 , 0 , 1 , -1)
		
		RMatrix = self.R_Matrix(x_grid)
		
		solution = self.LinearSolver(AMatrix,RMatrix)
		
		return solution

class Poisson1D_Mixed_BC_Solver2:
	def __init__(self, a , b , alpha , gamma , N , rho):
		
		# Extreme points x0 = a, xN = b
		self.a = a
		self.b = b
		
		# Boundary conditions:
		self.alpha = alpha # phi(0) = alpha
		self.gamma = gamma # phi'(N) = gamma
		
		# Number of mesh elements
		self.N = N
		
		# Density function
		self.rho = rho
		
		# Step size:
		self.h = self.b/self.N
	
	def grid_points(self,start_point, end_point, step_size):
		'''Construct the grid points'''
		return np.arange(start_point , end_point + step_size , step_size)
	
	def CalculateEigenSystem(self,A_matrix):
		EigenVals, EigenVec = np.linalg.eig(A_matrix)
		return EigenVals, EigenVec
	
	# We define a method to construct a tridiagonal matrix:
	def A_Matrix(self,N_Dim ,val1, val2, val3, k1 = 0, k2 = 1, k3 = -1):
		# The method constructs three diagonal matrices with k_i indicating the shift from the main diagonal
		''' val1,2,3: 1-D arrays
			k1,2,3: Diagonals where the arrays will be positioned'''
		
		tridiagonal_mat = np.diag(val1, k1) + np.diag(val2, k2) + np.diag(val3, k3) # Three main diagonals
		
		# Modify the last row of the matrix where the calculation of the derivative is done
		tridiagonal_mat[N_Dim-1][N_Dim-1] = 1 # Modify last element from last row and last column to 1
		
		A_mat = (1/self.h**2)*tridiagonal_mat # Multiply by the factor of 1/h^2
		
		return A_mat
	
	def R_Matrix(self,g_points):
		''' We construct the R matrix to solve the Ax=R system of equations. We define the function rho applied to the
		grid points and then replace the end points with the boundary conditions.'''
		
		matrix = self.rho(g_points[1:])
		matrix[0] = matrix[0] + self.alpha/self.h**2
		matrix[self.N - 1] = self.gamma/self.h
		return matrix
	
	def LinearSolver(self,A_matrix,R_matrix):
		''' We solve the linear system of equations A_matrix*X = R_matrix for X, then insert the boundary conditions
		at the start and end of the array'''
		
		sol = list(np.array(np.linalg.solve(A_matrix,R_matrix)))
		sol.insert(0,self.alpha)
		return sol
	
	def run(self):

		x_grid = self.grid_points(self.a , self.b , self.h)
		
		diagonal1 = 2*np.ones(self.N  , dtype = float) # Main diagonal must be of length N
		diagonal2 = -1*np.ones(self.N - 1 , dtype = float)
		diagonal3 = -1*np.ones(self.N - 1 ,dtype = float)
		AMatrix = self.A_Matrix(self.N , diagonal1 , diagonal2 , diagonal3 , 0 , 1 , -1)
		
		RMatrix = self.R_Matrix(x_grid)
		
		solution = self.LinearSolver(AMatrix,RMatrix)
		
		return solution

class Poisson1D_Periodic_BC_Solver1:
	def __init__(self, a , b , N , rho):
		
		# Extreme points x0 = a, xN = b
		self.a = a
		self.b = b
		
		# Number of mesh elements
		self.N = N
		
		# Density function
		self.rho = rho
		
		# Step size:
		self.h = self.b/self.N
	
	def grid_points(self,start_point, end_point, step_size):
		'''Construct the grid points'''
		return np.arange(start_point , end_point + step_size , step_size)
	
	def CalculateEigenSystem(self,A_matrix):
		EigenVals, EigenVec = np.linalg.eig(A_matrix)
		return EigenVals, EigenVec
	
	# We define a method to construct a tridiagonal matrix:
	def A_Matrix(self,N_Dim ,val1, val2, val3, k1=0, k2=1, k3=-1):
		# The method constructs three diagonal matrices with k_i indicating the shift from the main diagonal
		''' val1,2,3: 1-D arrays
			k1,2,3: Diagonals where the arrays will be positioned'''
		
		tridiagonal_mat = np.diag(val1, k1) + np.diag(val2, k2) + np.diag(val3, k3) # Three main diagonals
		
		# Modify the last row of the matrix where the calculation of the derivative is done
		tridiagonal_mat[0][N_Dim-2] = 1 # Modify last element from last row and last column to 1
		tridiagonal_mat[0][0] = 3
		
		A_mat = (1/self.h**2)*tridiagonal_mat # Multiply by the factor of 1/h^2
		
		return A_mat
	
	def R_Matrix(self,g_points):
		''' We construct the R matrix to solve the Ax=R system of equations. We define the function rho applied to the
		grid points and then replace the end points with the boundary conditions.'''
		
		if hasattr(self.rho, '__call__'):
			# rho is a function or callable
			matrix = self.rho(g_points[1:-1])
			matrix[0] = matrix[0] - self.rho(g_points[0])		
		else:
			# rho is an array
			matrix = self.rho[1:-1].copy()
			matrix[0] = matrix[0] - self.rho[0]

		return matrix
	
	def LinearSolver(self,A_matrix,R_matrix):
		''' We solve the linear system of equations A_matrix*X = R_matrix for X, then insert the boundary conditions
		at the start and end of the array'''
		
		sol = list(np.array(np.linalg.solve(A_matrix,R_matrix))) # Solve linear system
		sol.append(0) # left BC
		sol.insert(0,0) # periodic BC u[0] = u[N]
		return sol
	
	def run(self):

		x_grid = self.grid_points(self.a , self.b , self.h)

		diagonal1 = 2*np.ones(self.N -1  , dtype = float) # Main diagonal must be of length N
		diagonal2 = -1*np.ones(self.N - 2 , dtype = float)
		diagonal3 = -1*np.ones(self.N - 2 ,dtype = float)
		AMatrix = self.A_Matrix(self.N , diagonal1 , diagonal2 , diagonal3 , 0 , 1 , -1)
		
		RMatrix = self.R_Matrix(x_grid)
		
		solution = self.LinearSolver(AMatrix,RMatrix)
		
		return solution

class Poisson1D_Periodic_BC_Solver2:
	def __init__(self, a , b , N , rho):
		
		# Extreme points x0 = a, xN = b
		self.a = a
		self.b = b
		
		# Number of mesh elements
		self.N = N
		
		# Density function
		self.rho = rho
		
		# Step size:
		self.h = self.b/self.N
	
	def grid_points(self,start_point, end_point, step_size):
		'''Construct the grid points'''
		return np.arange(start_point , end_point + step_size , step_size)
	
	def CalculateEigenSystem(self,A_matrix):
		EigenVals, EigenVec = np.linalg.eig(A_matrix)
		return EigenVals, EigenVec
	
	# We define a method to construct a tridiagonal matrix:
	def A_Matrix(self,N_Dim ,val1, val2, val3, k1, k2, k3 , bottom_row , last_column):
		# The method constructs three diagonal matrices with k_i indicating the shift from the main diagonal
		''' val1,2,3: 1-D arrays
			k1,2,3: Diagonals where the arrays will be positioned'''
		
		Z_mat = np.zeros((N_Dim+1,N_Dim+1)) # Full matrix with zeros
		
		tridiagonal_mat =  np.diag(val1, k1) + np.diag(val2, k2) + np.diag(val3, k3) # Three main diagonals
		tridiagonal_mat[0][N_Dim-1] = -1 # Modify the element of the first column and last row with -1
		tridiagonal_mat[N_Dim-1][0] = -1 # Modify the element of the last column and first row with -1
		
		Z_mat[:N_Dim,:N_Dim] = tridiagonal_mat # Replace the N first columns and N first rows of Matrix with tridiagonal mat
		
		Z_mat[N_Dim,:N_Dim] = bottom_row # Replace bottom row with h**3
		Z_mat[:N_Dim,N_Dim] = last_column # Replace last column with h**2
		
		A_mat = (1/self.h**2)*Z_mat # Multiply with 1/h^2 factor
		
		return A_mat
	
	def R_Matrix(self,N_Dim,g_points):
		''' We construct the R matrix to solve the Ax=R system of equations. We define the function rho applied to the
		grid points and then replace the end points with the boundary conditions.'''
		
		matrix = np.zeros(N_Dim+1)
		matrix[1:-1] = self.rho(g_points[1:-1])
		
		return matrix
	
	def LinearSolver(self,A_matrix,R_matrix):
		''' We solve the linear system of equations A_matrix*X = R_matrix for X, then insert the boundary conditions
		at the start and end of the array'''
		
		sol = list(np.array(np.linalg.solve(A_matrix,R_matrix)))[:self.N]
		sol.append(sol[0])
#         sol.append(sol[0])
#         sol.append(0)
#         sol.insert(0,0)
		return sol
	
	def run(self):
		
		x_grid = self.grid_points(self.a , self.b , self.h)

		diagonal1 = 2*np.ones(self.N , dtype = float) # Main diagonal must be of length N
		diagonal2 = -1*np.ones(self.N - 1 , dtype = float)
		diagonal3 = -1*np.ones(self.N - 1 ,dtype = float)
		
		bot_row = (self.h**3) * np.ones(self.N)
		last_col = (self.h**2) * np.ones(self.N)
		
		AMatrix = self.A_Matrix(self.N , diagonal1 , diagonal2 , diagonal3 , 0 , 1 , -1 , bot_row,last_col)
		
		RMatrix = self.R_Matrix(self.N , x_grid)
		
		solution = self.LinearSolver(AMatrix , RMatrix)
		
		return solution