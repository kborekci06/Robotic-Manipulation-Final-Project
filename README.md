# Robotic-Manipulation-Final-Project
Final project for course MECH_ENG 449: Robotic Manipulation

# Project Overview:
This project implements motion planning and computed-torque control for a 6-DOF UR5 robot arm using the Modern Robotics (MR) library.
The system:
1.	Reads an input specification (input.txt) describing
    - Start and end end-effector configurations
    - Trajectory type (Screw or Cartesian, cubic or quintic)
    - Total trajectory duration
    - Initial joint configuration
    - Optional controller gains
    - Optional actuator torque limits
    - Optional timestep
2.	Generates a smooth end-effector trajectory in SE(3) using MR functions
3.	Uses numerical inverse kinematics at each time step to compute a joint-space reference trajectory
4.	Simulates closed-loop control using a PD + computed torque controller with joint damping
5.	Saves results to a set of CSV files and produces plots of;
    - Joint angles
    - Joint torques
    - End-effector error norms
6.	Exports a CoppeliaSim Scene2 CSV file to allow animated visualization

The project demonstrates three example scenarios illustrating:
	•	Good (critically damped) control
	•	Underdamped oscillatory behavior
	•	Torque-limited poor tracking performance

# How to Run the Code:
I ran the code in a anaconda virtual environment that has the following pip installed:
    - Python 3
    - numpy
    - matplotlib
    - The Modern Robotics Python library (modern_robotics.py)

In terminal:
conda activate v_env
python main.py


# Explanation of Examples:

Example 1 – Critically Damped Controller
Trajectory: Screw_Quintic
Gains: Kp = 16, Kd = 8 (critically damped)
Torque limits: None
Behavior:
    - Smooth, stable motion
    - Joint angles converge cleanly
    - Low torques after initial transient
    - End-effector error norms decay monotonically
This example shows good controller performance with accurate model based compensation.

Example 2 – Underdamped (Oscillatory) Controller
Trajectory: Screw_Quintic
Gains: Kp = 16, Kd = 2 (underdamped)
Torque limits: None
Behavior:
    - Noticeable overshoot and oscillations in joint angles
    - End-effector error norms oscillate before converging
    - Torques show ringing behavior
This shows how insufficient derivative damping Kd leads to oscillatory responses.

Example 3 – Torque-Limited Controller (Poor Tracking)
Trajectory: Screw_Quintic
Gains: Kp = 16, Kd = 8 (good controller, same as Example 1)
Torque limits: tau_max = 10 Nm
Behavior:
    - Torque saturation prevents the robot from generating sufficient force
    - End-effector error is larger than Examples 1 & 2
    - Joint trajectories lag behind reference and do not reach the goal by Tf
    - Torque plots show flat saturation regions
This example shows how low actuator torque limits prevent reaching end-effector configuration even with a well tuned controller.

