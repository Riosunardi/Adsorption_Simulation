import numpy as np
import matplotlib.pyplot as plt

# Constants
nu = 0.005  # convective velocity in m/s
c_A_in = 1  # inlet concentration in mol/m^3
c_B_in = 3  # inlet concentration in mol/m^3
L = 1  # column length in meters
time = 1000  # total time in seconds
epsilon = 0.3  # bed porosity
k_A = 0.1  # 1/s
K_A = 0.5  # m^3/mol
k_B = 0.0005  # 1/s
K_B = 0.0001  # m^3/mol

# Discretization parameters
dz = 0.01  # spatial step size
dt = 0.8 * dz / nu  # time step size

# Grid points
z_values = np.arange(0, L + dz, dz)
t_values = np.arange(0, time + dt, dt)

# Initialize concentration array
c_A = np.ones((len(t_values), len(z_values))) * 0
q_A = np.ones((len(t_values), len(z_values))) * 0
c_B = np.ones((len(t_values), len(z_values))) * 0
q_B = np.ones((len(t_values), len(z_values))) * 0

# Boundary conditions
# At the inlet (z = 0), the concentration is c_in
c_A[:, 0] = c_A_in
c_B[:, 0] = c_B_in
# At the outlet (z = L), the change in concentration over change in space is zero (dc/dz = 0)
c_A[:, -1] = c_A[:, -2]
c_B[:, -1] = c_B[:, -2]
q_A[:, -1] = q_A[:, -2]
q_B[:, -1] = q_B[:, -2]

# Define v to simplify the equation
v = nu * dt / dz

# Set r
r_A = np.zeros((len(t_values), len(z_values)))
r_B = np.zeros((len(t_values), len(z_values)))

# Solve using Van Leer flux limiter
for n in range(1, len(t_values) - 1):
    for i in range(1, len(z_values) - 1):

        r_A[n, i] = (c_A[n, i] - c_A[n, i-1] + 1e-10) / (c_A[n, i+1] - c_A[n, i] + 1e-10)
        r_B[n, i] = (c_B[n, i] - c_B[n, i-1] + 1e-10) / (c_B[n, i+1] - c_B[n, i] + 1e-10)

        idx_minus_half = int(i - 0.5)
        idx_plus_half = int(i + 0.5)

        phi_value_plus_A = (r_A[n, idx_plus_half] + np.abs(r_A[n, idx_plus_half])) / (1 + np.abs(r_A[n, idx_plus_half]))
        phi_value_minus_A = (r_A[n, idx_minus_half] + np.abs(r_A[n, idx_minus_half])) / (1 + np.abs(r_A[n, idx_minus_half]))

        phi_value_plus_B = (r_B[n, idx_plus_half] + np.abs(r_B[n, idx_plus_half])) / (1 + np.abs(r_B[n, idx_plus_half]))
        phi_value_minus_B = (r_B[n, idx_minus_half] + np.abs(r_B[n, idx_minus_half])) / (1 + np.abs(r_B[n, idx_minus_half]))

        q_star_A = (0.5 * K_A * c_A[n, i]) / (1 + K_A * c_A[n, i] + K_B * c_B[n, i])
        q_star_B = (98 * K_B * c_B[n, i]) / (1 + K_A * c_A[n, i] + K_B * c_B[n, i])

        q_A[n+1, i] = q_A[n, i] + ((k_A * dt) * (q_star_A - q_A[n, i]))
        q_B[n+1, i] = q_B[n, i] + ((k_B * dt) * (q_star_B - q_B[n, i]))

        c_A[n+1, i] = (c_A[n, i] - v * (c_A[n, i] - c_A[n, i-1]) - 0.5 * v * (1 - v) * (phi_value_plus_A * (c_A[n, i+1] - c_A[n, i]) - phi_value_minus_A * (c_A[n, i] - c_A[n, i-1]))) - (((1 - epsilon) / epsilon) * ((q_A[n+1, i] - q_A[n, i]) / dt))
        c_B[n+1, i] = (c_B[n, i] - v * (c_B[n, i] - c_B[n, i-1]) - 0.5 * v * (1 - v) * (phi_value_plus_B * (c_B[n, i+1] - c_B[n, i]) - phi_value_minus_B * (c_B[n, i] - c_B[n, i-1]))) - (((1 - epsilon) / epsilon) * ((q_B[n+1, i] - q_B[n, i]) / dt))
        
        q_A[n+1, 0] = q_A[n+1, 1]
        q_B[n+1, 0] = q_B[n+1, 1]

        # Print the time step
        if i == 1:
            # Number of time steps in the simulation
            num_steps = len(t_values) - 1
            print(f'Time step {n+1} completed out of {num_steps}')


# At the outlet (z = L), the change in concentration over change in space is zero (dc/dz = 0)
c_A[:, -1] = c_A[:, -2]
c_B[:, -1] = c_B[:, -2]
q_A[:, -1] = q_A[:, -2]
q_B[:, -1] = q_B[:, -2]

# Extract concentrations at the final time step
final_concentration_A = c_A[-1, :]
final_concentration_B = c_B[-1, :]
final_adsorbed_A = q_A[-1, :]
final_adsorbed_B = q_B[-1, :]

# # Plot results
# plt.figure(figsize=(10, 5))
# plt.plot(z_values, final_concentration_A, label='c_A')
# plt.plot(z_values, final_concentration_B, label='c_B')
# plt.xlabel('z (m)')
# plt.ylabel('c_i (mol/m^3)')
# plt.legend()
# plt.grid()
# plt.show()

# # Plot of the adsorbed quantities of A and B vs z
# plt.figure(figsize=(10, 5))
# plt.plot(z_values, final_adsorbed_A, label='q_A')
# plt.plot(z_values, final_adsorbed_B, label='q_B')
# plt.xlabel('z (m)')
# plt.ylabel('q (mol/m^3)')
# plt.legend()
# plt.grid()
# plt.show()

# Plot of composition at the outlet of the column vs time
plt.figure(figsize=(10, 10))
plt.plot(t_values, c_A[:, -1], label='$c_A$', linewidth=5, color='blue')
plt.plot(t_values, c_B[:, -1], label='$c_B$', linewidth=5, color='red')
plt.xlabel('$z\;(m)$', fontsize=20)
plt.ylabel('$c_{i}\;(mol/m^3) $', fontsize=20)
# plt.title('Fluid Concentration Profile at the Final Time (Advection Equation)', fontsize=35)
plt.legend(fontsize=15)
plt.grid(True)
# Change the font size for the ticks on the x and y axis
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
