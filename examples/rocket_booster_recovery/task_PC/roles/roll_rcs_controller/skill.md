# Roll RCS controller

Own only `RCS_x` roll stabilization. Explore roll-rate PD plus
deadband/hysteresis, PWPF, or constrained bang-bang with normal/emergency
torque and slew caps. Optimize stability, settling, switching, and pitch/yaw
coupling rather than fuel. `RCS_y=RCS_z=0` always. Use the frozen roll bank,
but remember roll-only evidence cannot parent.
