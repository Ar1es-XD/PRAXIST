# State and disturbance estimator

Use EKF/UKF, extended-state observer, or disturbance observer to produce
state/wind/drag/thrust-scale/mass-bias estimates for the analytic controller.
Perform an observability argument before combining mass, drag, and thrust bias.
No learned filter or adaptation from outcome labels is allowed. Keep the
estimator separable for clean ablation.
