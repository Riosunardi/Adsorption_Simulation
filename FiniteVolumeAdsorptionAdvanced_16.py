import numpy as np
import matplotlib.pyplot as plt

# Constants
nu = 0.0015  # convective velocity in m/s
c_in = 3.5  # inlet concentration in mol/m^3
L = 3  # column length in meters
T = 1000  # total time in seconds
epsilon = 0.3  # bed porosity
k_A = 0.05  # 1/s
K_A = 0.4  # m^3/mol


# Discretization parameters
dz = 0.01  # spatial step size
dt = 0.9 * dz / nu  # time step size

# Grid points
z_values = np.arange(0, L + dz, dz)
t_values = np.arange(0, T + dt, dt)

# Initialize concentration array
c = np.ones((len(t_values), len(z_values))) * 0.01
q = np.ones((len(t_values), len(z_values))) * 0.003

# Initial condition
# The column is empty at t = 0 (initially no solute is present)
c[0, :] = 0.1

# Boundary conditions
# At the inlet (z = 0), the concentration is c_in
c[:, 0] = c_in
# At the outlet (z = L), the change in concentration over change in space is zero (dc/dz = 0)
c[:, -1] = c[:, -2]
q[:, -1] = q[:, -2]

# Define v to simplify the equation
v = nu * dt / dz

# Set r
r = np.zeros((len(t_values), len(z_values)))

# Solve using Van Leer flux limiter
for n in range(1, len(t_values) - 1):
    for i in range(1, len(z_values) - 1):

        r[n, i] = (c[n, i] - c[n, i-1] + 1e-10) / (c[n, i+1] - c[n, i] + 1e-10)

        idx_minus_half = int(i - 0.5)
        idx_plus_half = int(i + 0.5)

        phi_value_plus = (r[n, idx_plus_half] + np.abs(r[n, idx_plus_half])) / (1 + np.abs(r[n, idx_plus_half]))
        phi_value_minus = (r[n, idx_minus_half] + np.abs(r[n, idx_minus_half])) / (1 + np.abs(r[n, idx_minus_half]))

        q_star = (K_A * c[n, i]) / (1 + K_A * c[n, i])

        q[n+1, i] = q[n, i] + ((k_A * dt) * (q_star - q[n, i]))

        c[n+1, i] = (c[n, i] - v * (c[n, i] - c[n, i-1]) - 0.5 * v * (1 - v) * (phi_value_plus * (c[n, i+1] - c[n, i]) - phi_value_minus * (c[n, i] - c[n, i-1]))) - (((1 - epsilon) / epsilon) * ((q[n+1, i] - q[n, i]) / dt))


# Extract concentrations at the final time step
final_concentration = c[-1, :]

# # Plot results
# plt.figure(figsize=(10, 6))
# plt.plot(z_values, c[-1, :], label=f't = {T} s', color='blue')
# plt.xlabel('z (m)')
# plt.ylabel('c_i (mol/m^3)')
# plt.title('Concentration Profile at the Final Time (Van Leer Limiter)')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Plot of adsorbed quantities of A vs time vs z
# plt.figure(figsize=(10, 6))
# plt.contourf(z_values, t_values, q, cmap='viridis')
# plt.xlabel('z (m)')
# plt.ylabel('t (s)')
# plt.title('Adsorbed Quantities of A vs Time vs z')
# plt.colorbar(label='q (mol/m^3)')
# plt.show()

# Plot of