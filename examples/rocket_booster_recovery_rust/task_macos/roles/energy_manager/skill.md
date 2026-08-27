# Energy manager

Own only time-to-go, vertical velocity corridor, ignition condition, and
throttle bounds. Build a mass-reserve feasibility cage using analytic braking
distance, bang-bang structure, or deterministic root solving. If lateral
tracking would violate predicted touchdown mass, protect vertical braking and
the strict >2% reserve gate first. Initial fuel is fixed at 7000 kg. A coast
segment is valid only if powered braking reaches the <=1 m/s first-contact
speed gate before a leg touches; never depend on gear damping. Do not redesign
lateral guidance or attitude control.
