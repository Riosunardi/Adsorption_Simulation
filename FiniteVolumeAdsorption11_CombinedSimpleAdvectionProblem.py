# Combining all the methods to solve the simple advection problem

# First Order Accurate Method:
# Upwind Difference Scheme for Finite Volume Method

# Second Order Accurate Methods:
# Lax-Wendroff Scheme for Finite Volume Method
# Beam-Warming Scheme for Finite Volume Method

# Higher Resolution Methods:
# Van Leer Flux Limiter Scheme for Finite Volume Method
# Superbee Flux Limiter Scheme for Finite Volume Method

##########################################################################################################

import numpy as np
import matplotlib.pyplot as plt
csfont = {'fontname':'Times New Roman'}

##########################################################################################################

# General Parameters
nu = 0.0015  # velocity in m/s
c_in = 1  # inlet concentration in mol/m^3
L = 1  # column length in meters
T = 300  # total time in seconds

dz = 0.002  # spatial step size
dt = 0.5 * dz / nu  # time step size

# Grid points
z_values = np.arange(0, L + dz, dz)
t_values = np.arange(0, T + dt, dt)

##########################################################################################################

# Upwind Scheme of Finite Difference Method

# Initialise concentration array
c = np.zeros((len(t_values), len(z_values)))

# Initial condition
# The column is empty at t = 0 (initially no solute is present)
c[0, :] = 0

# Boundary conditions
# At the inlet (z = 0), the concentration is c_in
c[:, 0] = c_in
# At the outlet (z = L), the change in concentration over change in space is zero (dc/dz = 0)
c[:, -1] = c[:, -2]

# Solving the equation
for n in range(1, len(t_values)-1):
    for i in range(1, len(z_values)):
        c[n+1, i] = c[n, i] - nu * dt / dz * (c[n, i] - c[n, i-1])

##########################################################################################################
        
# Lax-Wendroff Scheme for Finite Volume Method
        
# Initialise concentration array
d = np.zeros((len(t_values), len(z_values)))

# Initial condition
# The column is empty at t = 0 (initially no solute is present)
d[0, :] = 0

#Boundary conditions
# At the inlet (z = 0), the concentration is c_in
d[:, 0] = c_in
# At the outlet (z = L), the change in concentration over change in space is zero (dc/dz = 0)
d[:, -1] = d[:, -2]

# Solve using Lax-Wendroff finite volume method
# The concentartion at the next time step is calculated using the concentration at the current time step
# The spatial derivative is approximated using a central difference scheme
for n in range(1, len(t_values) - 1):
    for i in range(1, len(z_values) - 1):
        d[n+1, i] = (d[n, i] - nu * dt / (2 * dz) * (d[n, i+1] - d[n, i-1]) +
                     nu**2 * dt**2 / (2 * dz**2) * (d[n, i+1] - 2 * d[n, i] + d[n, i-1]))
        
##########################################################################################################
        
# Beam-Warming Scheme for Finite Volume Method
        
# Initialise concentration array
e = np.zeros((len(t_values), len(z_values)))

# Initial condition
# The column is empty at t = 0 (initially no solute is present)
e[0, :] = 0

#Boundary conditions
# At the inlet (z = 0), the concentration is c_in
e[:, 0] = c_in
e[:, 1] = c_in
e[:, 2] = c_in

# At the outlet (z = L), the change in concentration over change in space is zero (dc/dz = 0)
e[:, -1] = e[:, -2]

# Solve using Beam-Warming method
for n in range(1, len(t_values) - 1):
    for i in range(2, len(z_values) - 1):  # Notice the starting index (i=2)
        e[n+1, i] = (e[n, i] - nu * dt / (2 * dz) * (3*e[n, i] - 4*e[n, i-1] + e[n, i-2]) +
                     nu**2 * dt**2 / (2 * dz**2) * (e[n, i] - 2*e[n, i-1] + e[n, i-2]))
        

##########################################################################################################

# Van Leer Flux Limiter Scheme for Finite Volume Method
        
# Initialize concentration array
f = np.zeros((len(t_values), len(z_values)))

# Initial condition
# The column is empty at t = 0 (initially no solute is present)
f[0, :] = 0

# Boundary conditions
# At the inlet (z = 0), the concentration is c_in
f[:, 0] = c_in
# At the outlet (z = L), the change in concentration over change in space is zero (dc/dz = 0)
f[:, -1] = f[:, -2]

# Define v to simplify the equation
v = nu * dt / dz

# Set r
r = np.zeros((len(t_values), len(z_values)))

# Solve using Van Leer flux limiter
for n in range(1, len(t_values) - 1):
    for i in range(1, len(z_values) - 1):

        r[n, i] = (f[n, i] - f[n, i-1] + 1e-10) / (f[n, i+1] - f[n, i] + 1e-10)

        idx_minus_half = int(i - 0.5)
        idx_plus_half = int(i + 0.5)

        phi_value_plus = (r[n, idx_plus_half] + np.abs(r[n, idx_plus_half])) / (1 + np.abs(r[n, idx_plus_half]))
        phi_value_minus = (r[n, idx_minus_half] + np.abs(r[n, idx_minus_half])) / (1 + np.abs(r[n, idx_minus_half]))

        f[n+1, i] = (f[n, i] - v * (f[n, i] - f[n, i-1]) - 0.5 * v * (1 - v) * (phi_value_plus * (f[n, i+1] - f[n, i]) - phi_value_minus * (f[n, i] - f[n, i-1])))

##########################################################################################################
        
# Superbee Flux Limiter Scheme for Finite Volume Method
        
# Initialize concentration array
g = np.zeros((len(t_values), len(z_values)))

# Initial condition
# The column is empty at t = 0 (initially no solute is present)
g[0, :] = 0

# Boundary conditions
# At the inlet (z = 0), the concentration is c_in
g[:, 0] = c_in
# At the outlet (z = L), the change in concentration over change in space is zero (dc/dz = 0)
g[:, -1] = g[:, -2]

# Set r
r_superbee = np.zeros((len(t_values), len(z_values)))

# Solve using Superbee flux limiter
for n in range(1, len(t_values) - 1):
    for i in range(1, len(z_values) - 1):

        r_superbee[n, i] = (g[n, i] - g[n, i-1] + 1e-10) / (g[n, i+1] - g[n, i] + 1e-10)

        idx_minus_half_superbee = int(i - 0.5)
        idx_plus_half_superbee = int(i + 0.5)

        phi_value_plus_superbee = max(0, min(2*r_superbee[n, idx_plus_half_superbee], 1, min(r_superbee[n, idx_plus_half_superbee], 2)))
        phi_value_minus_superbee = max(0, min(2*r_superbee[n, idx_minus_half_superbee], 1, min(r_superbee[n, idx_minus_half_superbee], 2)))

        g[n+1, i] = (g[n, i] - v * (g[n, i] - g[n, i-1]) - 0.5 * v * (1 - v) * (phi_value_plus_superbee * (g[n, i+1] - g[n, i]) - phi_value_minus_superbee * (g[n, i] - g[n, i-1])))

##########################################################################################################
        

# Plot on the same figure
# c vs z at the final time and d vs z at the final time

plt.figure(figsize=(10, 10))
plt.plot(z_values, c[-1, :], label='Upwind Scheme', color='red', linewidth=5)
plt.plot(z_values, d[-1, :], label='Lax-Wendroff Scheme', color='green', linewidth=5)
plt.plot(z_values, e[-1, :], label='Beam-Warming Scheme', color='orange', linewidth=5)
plt.plot(z_values, g[-1, :], label='Superbee Flux Limiter Scheme', color='brown', linewidth=5)
plt.plot(z_values, f[-1, :], label='Van Leer Flux Limiter Scheme', color='blue', linewidth=5)
plt.xlabel('$z\;(m)$', fontsize=20)
plt.ylabel('$c_{i}\;(mol/m^3) $', fontsize=20)
# plt.title('Fluid Concentration Profile at the Final Time (Advection Equation)', fontsize=35)
plt.legend(fontsize=15)
plt.grid(True)
# Change the font size for the ticks on the x and y axis
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()



