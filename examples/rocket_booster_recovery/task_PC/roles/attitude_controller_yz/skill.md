# Attitude controller YZ

Accept desired thrust direction/angular-rate reference and output pitch/yaw
moment demand only. Use SO(3) PD, LQR/LQG, or loop shaping scheduled by mass,
inertia, thrust, dynamic pressure, and phase. Do not modify guidance and never
emit or allocate any RCS channel.
