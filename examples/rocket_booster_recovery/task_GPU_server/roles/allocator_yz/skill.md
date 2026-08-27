# Allocator YZ

Own only constrained allocation of pitch/yaw moment demand to gimbal_y/z and
grid_y/z. Use a small QP or weighted least squares with amplitude, rate,
dynamic-pressure authority, moment residual, smoothness, and gimbal translation
reserve. `grid_roll=0`; RCS must never enter the y-z allocation matrix.
