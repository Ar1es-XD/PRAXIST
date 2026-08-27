pub type Vec2 = [f32; 2];
pub type Vec3 = [f32; 3];
pub type Vec4 = [f32; 4];
pub type Mat3 = [[f32; 3]; 3];

#[inline(always)]
pub fn clip(x: f32, low: f32, high: f32) -> f32 {
    x.max(low).min(high)
}

#[inline(always)]
pub fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline(always)]
pub fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline(always)]
pub fn scale(a: Vec3, s: f32) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline(always)]
pub fn mul(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}

#[inline(always)]
pub fn dot(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline(always)]
pub fn norm(a: Vec3) -> f32 {
    dot(a, a).sqrt()
}

#[inline(always)]
pub fn safe_norm(a: Vec3) -> f32 {
    (dot(a, a) + 1.0e-8).sqrt()
}

#[inline(always)]
pub fn safe_norm2(a: Vec2) -> f32 {
    (a[0] * a[0] + a[1] * a[1] + 1.0e-8).sqrt()
}

#[inline(always)]
pub fn unit(a: Vec3) -> Vec3 {
    scale(a, 1.0 / safe_norm(a))
}

#[inline(always)]
pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline(always)]
pub fn mat_vec(m: Mat3, v: Vec3) -> Vec3 {
    [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}

#[inline(always)]
pub fn mat_t_vec(m: Mat3, v: Vec3) -> Vec3 {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

#[inline(always)]
pub fn quat_mul(a: Vec4, b: Vec4) -> Vec4 {
    let [aw, ax, ay, az] = a;
    let [bw, bx, by, bz] = b;
    [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]
}

#[inline(always)]
pub fn normalize_quat(mut q: Vec4) -> Vec4 {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        .sqrt()
        .max(1.0e-8);
    q[0] /= n;
    q[1] /= n;
    q[2] /= n;
    q[3] /= n;
    q
}

#[inline(always)]
pub fn controller_normalize_quat(mut q: Vec4) -> Vec4 {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3] + 1.0e-8).sqrt();
    q[0] /= n;
    q[1] /= n;
    q[2] /= n;
    q[3] /= n;
    q
}

#[inline(always)]
pub fn quat_to_rot(q: Vec4) -> Mat3 {
    let [q0, q1, q2, q3] = normalize_quat(q);
    [
        [
            q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3,
            2.0 * (q1 * q2 - q0 * q3),
            2.0 * (q1 * q3 + q0 * q2),
        ],
        [
            2.0 * (q1 * q2 + q0 * q3),
            q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3,
            2.0 * (q2 * q3 - q0 * q1),
        ],
        [
            2.0 * (q1 * q3 - q0 * q2),
            2.0 * (q2 * q3 + q0 * q1),
            q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3,
        ],
    ]
}

#[inline(always)]
pub fn controller_quat_to_rot(q: Vec4) -> Mat3 {
    let [q0, q1, q2, q3] = controller_normalize_quat(q);
    [
        [
            q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3,
            2.0 * (q1 * q2 - q0 * q3),
            2.0 * (q1 * q3 + q0 * q2),
        ],
        [
            2.0 * (q1 * q2 + q0 * q3),
            q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3,
            2.0 * (q2 * q3 - q0 * q1),
        ],
        [
            2.0 * (q1 * q3 - q0 * q2),
            2.0 * (q2 * q3 + q0 * q1),
            q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3,
        ],
    ]
}

#[inline(always)]
pub fn integrate_quat_exact(q: Vec4, omega: Vec3, dt: f32) -> Vec4 {
    let n = (dot(omega, omega) + 1.0e-12).sqrt();
    let half = 0.5 * n * dt;
    let c = half.cos();
    let s = half.sin();
    let dq = [c, s * omega[0] / n, s * omega[1] / n, s * omega[2] / n];
    normalize_quat(quat_mul(dq, q))
}

#[inline(always)]
pub fn gimbal_rot(dy: f32, dz: f32) -> Mat3 {
    let (sy, cy) = dy.sin_cos();
    let (sz, cz) = dz.sin_cos();
    [
        [cy * cz, -sy, -cy * sz],
        [sy * cz, cy, -sy * sz],
        [sz, 0.0, cz],
    ]
}

#[inline(always)]
pub fn all_finite<const N: usize>(x: &[f32; N]) -> bool {
    x.iter().all(|v| v.is_finite())
}
