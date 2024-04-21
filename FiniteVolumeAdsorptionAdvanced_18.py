import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# System: Nitrogen and Carbon Dioxide adsorption in a packed bed column
# The inlet gas is 15% Carbon Dioxide and 85% Nitrogen (y_A = 0.15, y_B = 0.85)
# Adsorption isotherms: Competitive Langmuir isotherms
# The adsorbent is Molecular Sieve Type 5A

# The pressure is 101325 Pa, Temperature is 298 K and R = 8.314 J/(mol*K)
# PV = nRT
# n/V = P/RT = 101325/(8.314*298) = 40.7 mol/m^3 = c_total
# c_A = 0.15 * 40.7 = 6.105 mol/m^3
# c_B = 0.85 * 40.7 = 34.595 mol/m^3

# Constants
nu = 0.5  # convective velocity in m/s
c_A_in = 6.105  # inlet concentration in mol/m^3
c_B_in = 34.595  # inlet concentration in mol/m^3
L = 0.5  # column length in meters
time = 80  # total time in seconds
epsilon = 0.35  # bed porosity
k_A = 1.5  # 1/s
k_B = 0.5  # 1/s

R = 8.314  # J/(mol*K)
T = 298  # K
r_p = 1e-3  # m
viscosity = 1.72e-5  # Pa*s

deltaU_A = 31.19  # kJ/mol
deltaU_B = 16.38  # kJ/mol

b_A0 = 2.5e-6  # m^3/mol
b_B0 = 2.7e-6  # m^3/mol

q_sA = 4.39*1000 # mol/m^3
q_sB = 4.39*1000 # mol/m^3
# These were multiplied by 1000 to convert from mol/kg to mol/m^3

b_iA = b_A0 * np.exp(deltaU_A / (R * T))
b_iB = b_B0 * np.exp(deltaU_B / (R * T))

D_m = 1.6e-5  # m^2/s # Molecular diffusivity
tau = 3 # Tortuosity factor
D_p = D_m / tau # m^2/s # Effective macropore diffusivity

# Discretisation parameters
dz = 0.01  # spatial step size
dt = 0.9 * dz / nu  # time step size

# Grid points
z_values = np.arange(0, L + dz, dz)
t_values = np.arange(0, time + dt, dt)

# Initialise arrays
c_A = np.ones((len(t_values), len(z_values))) * 0.001
q_A = np.ones((len(t_values), len(z_values))) * 0.0005
c_B = np.ones((len(t_values), len(z_values))) * 0.002
q_B = np.ones((len(t_values), len(z_values))) * 0.0001
P = np.ones((len(t_values), len(z_values))) * 101325  # Pa
u = np.ones((len(t_values), len(z_values))) * 0.01  # m/s
v = np.zeros((len(t_values), len(z_values))) * 0.001  # m/s
alpha_A = np.ones((len(t_values), len(z_values))) * 0.0001
alpha_B = np.ones((len(t_values), len(z_values))) * 0.0001
c_total = np.zeros((len(t_values), len(z_values)))
y_A = np.zeros((len(t_values), len(z_values)))
y_B = np.zeros((len(t_values), len(z_values)))
H = np.zeros((len(t_values), len(z_values)))
t_breakthrough = np.zeros((len(t_values), len(z_values)))


# Boundary conditions
# At the inlet (z = 0), the concentration is c_in
c_A[:, 0] = c_A_in
c_B[:, 0] = c_B_in
u[:, 0] = nu
# Inlet pressure is calculated using a known inlet velocity and the Darcy equation
P[:, 0] = P[:, 1] +  (nu * (dz/2) /((4/150) * (r_p**2) * ((epsilon) /(1-epsilon))**2 * (1/viscosity)))
# At the outlet (z = L), the change in concentration over change in space is zero (dc/dz = 0)
c_A[:, -1] = c_A[:, -2]
c_B[:, -1] = c_B[:, -2]
q_A[:, -1] = q_A[:, -2]
q_B[:, -1] = q_B[:, -2]
# The pressure at the outlet is 1 bar or equivalent to 101325 Pa
P[:, -1] = 101325

# Set r
r_A_plus = np.zeros((len(t_values), len(z_values)))
r_A_minus = np.zeros((len(t_values), len(z_values)))
r_B_plus = np.zeros((len(t_values), len(z_values)))
r_B_minus = np.zeros((len(t_values), len(z_values)))
r_P_plus = np.zeros((len(t_values), len(z_values)))
r_P_minus = np.zeros((len(t_values), len(z_values)))

# Solve using Van Leer flux limiter
for n in range(1, len(t_values) - 1):

    P[n+1, 0] = P[n, 1] +  ((u[1, 1] * (dz/2)) / ((4/150) * (r_p**2) * ((epsilon) /(1-epsilon))**2 * (1/(viscosity))))

    for i in range(1, len(z_values) - 1):

        # r_A[n, i] = (c_A[n, i] - c_A[n, i-1] + 1e-10) / (c_A[n, i+1] - c_A[n, i] + 1e-10)
        # r_B[n, i] = (c_B[n, i] - c_B[n, i-1] + 1e-10) / (c_B[n, i+1] - c_B[n, i] + 1e-10)

        # idx_minus_half = int(i - 0.5)
        # idx_plus_half = int(i + 0.5)

        # phi_value_plus_A = (r_A[n, idx_plus_half] + np.abs(r_A[n, idx_plus_half])) / (1 + np.abs(r_A[n, idx_plus_half]))
        # phi_value_minus_A = (r_A[n, idx_minus_half] + np.abs(r_A[n, idx_minus_half])) / (1 + np.abs(r_A[n, idx_minus_half]))

        # phi_value_plus_B = (r_B[n, idx_plus_half] + np.abs(r_B[n, idx_plus_half])) / (1 + np.abs(r_B[n, idx_plus_half]))
        # phi_value_minus_B = (r_B[n, idx_minus_half] + np.abs(r_B[n, idx_minus_half])) / (1 + np.abs(r_B[n, idx_minus_half]))

        r_A_plus[n, i] = (c_A[n, i] - c_A[n, i-1] + 1e-10) / (c_A[n, i+1] - c_A[n, i] + 1e-10)
        r_A_minus[n, i] = (c_A[n, i-1] - c_A[n, i-2] + 1e-10) / (c_A[n, i] - c_A[n, i-1] + 1e-10)

        r_B_plus[n, i] = (c_B[n, i] - c_B[n, i-1] + 1e-10) / (c_B[n, i+1] - c_B[n, i] + 1e-10)
        r_B_minus[n, i] = (c_B[n, i-1] - c_B[n, i-2] + 1e-10) / (c_B[n, i] - c_B[n, i-1] + 1e-10)

        r_P_plus[n, i] = (P[n, i] - P[n, i-1] + 1e-10) / (P[n, i+1] - P[n, i] + 1e-10)
        r_P_minus[n, i] = (P[n, i-1] - P[n, i-2] + 1e-10) / (P[n, i] - P[n, i-1] + 1e-10)

        phi_value_plus_A = (r_A_plus[n, i] + np.abs(r_A_plus[n, i])) / (1 + np.abs(r_A_plus[n, i]))
        phi_value_minus_A = (r_A_minus[n, i] + np.abs(r_A_minus[n, i])) / (1 + np.abs(r_A_minus[n, i]))

        phi_value_plus_B = (r_B_plus[n, i] + np.abs(r_B_plus[n, i])) / (1 + np.abs(r_B_plus[n, i]))
        phi_value_minus_B = (r_B_minus[n, i] + np.abs(r_B_minus[n, i])) / (1 + np.abs(r_B_minus[n, i]))

        phi_value_plus_P = (r_P_plus[n, i] + np.abs(r_P_plus[n, i])) / (1 + np.abs(r_P_plus[n, i]))
        phi_value_minus_P = (r_P_minus[n, i] + np.abs(r_P_minus[n, i])) / (1 + np.abs(r_P_minus[n, i]))

        q_star_A = (q_sA * b_iA * c_A[n, i]) / (1 + (b_iA * c_A[n, i]) + (b_iB * c_B[n, i]))
        q_star_B = (q_sB * b_iB * c_B[n, i]) / (1 + (b_iA * c_A[n, i]) + (b_iB * c_B[n, i]))

        alpha_A[n, i] = (c_A[n, i] / q_star_A) * ((15 * epsilon * D_p) / (r_p**2)) * (L / nu)
        alpha_B[n, i] = (c_B[n, i] / q_star_B) * ((15 * epsilon * D_p) / (r_p**2)) * (L / nu)

        q_A[n+1, i] = q_A[n, i] + ((k_A * dt) * (q_star_A - q_A[n, i]))
        q_B[n+1, i] = q_B[n, i] + ((k_B * dt) * (q_star_B - q_B[n, i]))

        v[n, i] = u[n, i] * dt / dz

        P[n+1, i] = P[n, i] - (v[n, i] * (P[n, i] - P[n, i-1]) - 0.5 * v[n, i] * (1 - v[n, i]) * (phi_value_plus_P * (P[n, i+1] - P[n, i]) - phi_value_minus_P * (P[n, i] - P[n, i-1]))) - ((R * T / P[n, i]) * ((1 - epsilon) / epsilon) * (((q_A[n+1, i] - q_A[n, i]) + (q_B[n+1, i] - q_B[n, i]))))
        
        u[n, i] = (4 / (150 * viscosity)) * (epsilon / (1 - epsilon))**2 * (r_p**2) * (-(P[n, i] - P[n, i-1]) / dz)

        c_A[n+1, i] = (c_A[n, i] - v[n, i] * (c_A[n, i] - c_A[n, i-1]) - 0.5 * v[n, i] * (1 - v[n, i]) * (phi_value_plus_A * (c_A[n, i+1] - c_A[n, i]) - phi_value_minus_A * (c_A[n, i] - c_A[n, i-1]))) - (((1 - epsilon) / epsilon) * ((q_A[n+1, i] - q_A[n, i])))
        c_B[n+1, i] = (c_B[n, i] - v[n, i] * (c_B[n, i] - c_B[n, i-1]) - 0.5 * v[n, i] * (1 - v[n, i]) * (phi_value_plus_B * (c_B[n, i+1] - c_B[n, i]) - phi_value_minus_B * (c_B[n, i] - c_B[n, i-1]))) - (((1 - epsilon) / epsilon) * ((q_B[n+1, i] - q_B[n, i])))

        q_A[n+1, 0] = q_A[n+1, 1]
        q_B[n+1, 0] = q_B[n+1, 1]

        # H[n+1, i] = (((1 + (b_iA * c_A[n+1, i]) + (b_iB * c_B[n+1, i])) * (q_sA * b_iA)) - ((q_sA * b_iA * c_A[n+1, i]) * (1 + b_iA + (b_iB * c_B[n+1, i])))) / ((1 + (b_iA * c_A[n+1, i]) + (b_iB * c_B[n+1, i]))**2)

        # t_breakthrough[n+1, i] = (L / nu) * (epsilon + (1 - epsilon)) * H[n+1, i]


        # Print the time step
        if i == 1:
            # Number of time steps in the simulation
            num_steps = len(t_values) - 1
            print(f'Time step {n+1} completed out of {num_steps}')
                  
# P[:, 0] = P[:, 1] +  (nu * (dz/2) /((4/150) * (r_p**2) * ((epsilon) /(1-epsilon))**2 * (1/viscosity)))
# P[:, -1] = P[:, -2]
c_A[:, -1] = c_A[:, -2]
c_B[:, -1] = c_B[:, -2]
q_A[:, -1] = q_A[:, -2]
q_B[:, -1] = q_B[:, -2]


# Converting the concentration values to fluid fractions
c_total = c_A + c_B
y_A = c_A / c_total
y_B = c_B / c_total

# # Plot concentration profile at the final time
# plt.figure(figsize=(10, 5))
# plt.plot(z_values, c_A[-1, :], label='c_A')
# plt.plot(z_values, c_B[-1, :], label='c_B')
# plt.xlabel('z (m)')
# plt.ylabel('c_i (mol/m^3)')
# plt.legend()
# plt.grid()
# plt.show()

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

# # Plot the two above plots side by side
# fig, axs = plt.subplots(1, 2, figsize=(15, 5))
# axs[0].plot(z_values, final_concentration_A, label='c_A')
# axs[0].plot(z_values, final_concentration_B, label='c_B')
# axs[0].set_xlabel('z (m)')
# axs[0].set_ylabel('c_i (mol/m^3)')
# axs[0].legend()
# axs[0].grid()
# axs[1].plot(z_values, final_adsorbed_A, label='q_A')
# axs[1].plot(z_values, final_adsorbed_B, label='q_B')
# axs[1].set_xlabel('z (m)')
# axs[1].set_ylabel('q (mol/m^3)')
# axs[1].legend()
# axs[1].grid()
# plt.show()

# # Plot of pressure profile
# plt.figure(figsize=(10, 5))
# plt.plot(z_values, P[-1, :])
# plt.xlabel('z (m)')
# plt.ylabel('P (Pa)')
# plt.grid()
# plt.show()

# # Plot of adsorbed quantities of A and B vs z
# plt.figure(figsize=(10, 5))
# plt.plot(z_values, q_A[-1, :], label='q_A')
# plt.plot(z_values, q_B[-1, :], label='q_B')
# plt.xlabel('z (m)')
# plt.ylabel('q (mol/kg)')
# plt.legend()
# plt.grid()
# plt.show()

# # Plot concentration profile at the final time and adsorbed quantities of A and B vs z side by side
# fig, axs = plt.subplots(1, 2, figsize=(15, 5))
# axs[0].plot(z_values, c_A[-1, :], label='c_A')
# axs[0].plot(z_values, c_B[-1, :], label='c_B')
# axs[0].set_xlabel('z (m)')
# axs[0].set_ylabel('c_i (mol/m^3)')
# axs[0].legend()
# axs[0].grid()
# axs[1].plot(z_values, q_A[-1, :], label='q_A')
# axs[1].plot(z_values, q_B[-1, :], label='q_B')
# axs[1].set_xlabel('z (m)')
# axs[1].set_ylabel('q (mol/kg)')
# axs[1].legend()
# axs[1].grid()
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

# # Plot of fluid fractions at the outlet of the column vs time
# plt.figure(figsize=(10, 5))
# plt.plot(t_values, y_A[:, -1], label='y_A')
# plt.plot(t_values, y_B[:, -1], label='y_B')
# plt.xlabel('Time (s)')
# plt.ylabel('y_i')
# plt.legend()
# plt.grid()
# plt.show()


# # 3D plot of breakthrough time vs z vs time
# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(111, projection='3d')
# X, Y = np.meshgrid(z_values, t_values)
# ax.plot_surface(X, Y, t_breakthrough, cmap='viridis')
# ax.set_xlabel('z (m)')
# ax.set_ylabel('Time (s)')
# ax.set_zlabel('Breakthrough Time (s)')
# plt.show()








