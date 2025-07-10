module Models

using ControlSystemsBase
using LinearAlgebra

export benchmarks

# Resistor-capacitor network
sys_rcn = let
	r_1 = 100000
	r_2 = 500000
	r_3 = 200000
	c_1 = 0.000002
	c_2 = 0.000010
	A = [-1/c_1 * (1/r_1 + 1/r_2)  1/(r_2*c_1)
	     1/(r_2*c_2)               -1/c_2 * (1/r_2 + 1/r_3)]
	B = [1/(r_1*c_1)
	     1/(r_3*c_2)]
	# C = [1  -1]
    C = I
	D = 0

	ss(A, B, C, D)
end

# F1-tenth car
sys_f1t = let 
    v = 6.5
    L = 0.3302
    d = 1.5
    A = [0 v ; 0 0]
    B = [0; v/L]
    C = [1 0]
    D = 0

    ss(A, B, C, D)
end

# DC motor
sys_dcm = let
    A = [-10 1; -0.02 -2]
    B = [0; 2]
    C = [1 0]
    D = 0

    ss(A, B, C, D)
end

# Car suspension system
sys_css = let
    A = [0. 1 0 0; -8 -4 8 4; 0 0 0 1; 80 40 -160 -60]
    B = [0; 80; 20; -1120]
    C = [1 0 0 0]
    D = 0

    ss(A, B, C, D)
end

# Electronic wedge brake
sys_ewb = let
    A = [0 1; 8.3951e3 0]
    B = [0; 4.0451]
    C = [7.9920e3 0]
    D = 0

    ss(A, B, C, D)
end

# Cruise control 1
sys_cc1 = let
    A = -0.05
    B = 0.01
    C = 1
    D = 0

    ss(A, B, C, D)
end

# Cruise control 2
sys_cc2 = let
    A = [0 1 0; 0 0 1; -6.0476 -5.2856 -0.238]
    B = [0; 0; 2.4767]
    C = [1 0 0]
    D = 0

    ss(A, B, C, D)
end

sys_mpc = ss(tf([3, 1],[1, 0.6, 1]))

# Adaptive cruise control with lane keeping (circular lane)
sys_acc_lk = let
    # Vehicle parameters
    m = 1500.0      # Vehicle mass (kg)
    Iz = 2500.0     # Yaw moment of inertia (kg⋅m²)
    lf = 1.2        # Distance from CG to front axle (m)
    lr = 1.6        # Distance from CG to rear axle (m)
    Cf = 80000.0    # Front tire cornering stiffness (N/rad)
    Cr = 100000.0   # Rear tire cornering stiffness (N/rad)
    vx = 20.0       # Nominal longitudinal velocity (m/s)

    # State matrix A (8x8)
    # States: [y, ψ, ẏ, ψ̇, s, vx, d, vrel]
    # y:    Lateral deviation from lane center (meters)
    # ψ:    Heading angle error (radians) - angle between vehicle heading and lane direction
    # ẏ:    Lateral velocity (m/s)
    # ψ̇:    Yaw rate (rad/s) - how fast the vehicle is rotating
    # s:    Longitudinal position (meters)
    # vx:   Longitudinal velocity (m/s)
    # d:    Distance to vehicle ahead (meters)
    # vrel: lead vehicle velocity (m/s)
    A = [0    vx   1    0    0    0    0    0;
        0    0    0    1    0    0    0    0;
        0    0    -2*(Cf+Cr)/(m*vx)    -vx-2*(lf*Cf-lr*Cr)/(m*vx)    0    2*(Cf+Cr)/m    0    0;
        0    0    -2*(lf*Cf-lr*Cr)/(Iz*vx)    -2*(lf^2*Cf+lr^2*Cr)/(Iz*vx)    0    2*(lf*Cf-lr*Cr)/Iz    0    0;
        0    0    0    0    0    1    0    0;
        0    0    0    0    0    0    0    0;
        0    0    0    0    0    -1   0    1;
        0    0    0    0    0    0    0    0]

    # Control matrix B (8x4)
    # Controls: [δf, ax, δr, jx]
    # δf:   Front wheel steering angle (radians) - how much the front wheels are turned
    # ax:   Longitudinal acceleration (m/s²) - acceleration/braking command
    # δr:   Rear wheel steering angle (radians) - how much the rear wheels are turned (if equipped)
    # jx:   Longitudinal jerk (m/s³) - rate of change of acceleration (for smooth control)
    B = [0    0    0    0;
        0    0    0    0;
        2*Cf/m    0    2*Cr/m    0;
        2*lf*Cf/Iz    0    -2*lr*Cr/Iz    0;
        0    0    0    0;
        0    1    0    0;
        0    0    0    0;
        0    -1   0    0]
    
    # C = I
    C = [1. 0 0 0 0 0 0 0
         0 0 0 0 0 0 1 0]  # Full state output
    D = zeros(2, 4)
    
    ss(A, B, C, D)
end

benchmarks = Dict([
    :RC => sys_rcn,
    :F1 => sys_f1t,
    :DC => sys_dcm,
    :CS => sys_css,
    :EW => sys_ewb,
    :C1 => sys_cc1,
    :CC => sys_cc2,
    :MPC => sys_mpc,
    :ACCLK => sys_acc_lk
])

end
