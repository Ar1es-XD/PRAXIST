# Trajectory guidance

Generate smooth position, velocity, and thrust-direction references; never
direct actuator commands. Explore warm-started low-frequency SCP/SOCP/QP-MPC
or improved reachability-aware analytic guidance with explicit thrust, tilt,
gimbal, glide-cone, and terminal constraints. On solver failure, fall back to
the energy safety cage. Report solver failure and worst latency if applicable.
