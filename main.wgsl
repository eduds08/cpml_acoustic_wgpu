struct InfoInt {
    grid_size_z: i32,
    grid_size_x: i32,
    source_z: i32,
    source_x: i32,
    i: i32,
};

struct InfoFloat {
    dz: f32,
    dx: f32,
    dt: f32,
};

@group(0) @binding(0) // Info Int
var<storage,read_write> infoI32: InfoInt;

@group(0) @binding(1) // Info Float
var<storage,read> infoF32: InfoFloat;

@group(0) @binding(2) // source term
var<storage,read> source: array<f32>;

@group(0) @binding(3) // pressure field future
var<storage,read_write> p_future: array<f32>;

@group(0) @binding(4) // pressure field present
var<storage,read_write> p_present: array<f32>;

@group(0) @binding(5) // pressure field past
var<storage,read_write> p_past: array<f32>;

@group(0) @binding(6) // cfl
var<storage,read> cfl: array<f32>;

@group(0) @binding(7)
var<storage,read_write> phi_z: array<f32>;

@group(0) @binding(8)
var<storage,read_write> phi_x: array<f32>;

@group(0) @binding(9)
var<storage,read_write> absorption_z: array<f32>;

@group(0) @binding(10)
var<storage,read_write> absorption_x: array<f32>;

@group(0) @binding(11)
var<storage,read_write> is_z_absorption_int32: array<i32>;

@group(0) @binding(12)
var<storage,read_write> is_x_absorption_int32: array<i32>;

@group(0) @binding(13)
var<storage,read_write> psi_z: array<f32>;

@group(0) @binding(14)
var<storage,read_write> psi_x: array<f32>;

@group(0) @binding(15)
var<storage,read_write> z_diff_1: array<f32>;

@group(0) @binding(16)
var<storage,read_write> z_diff_2: array<f32>;

@group(0) @binding(17)
var<storage,read_write> x_diff_1: array<f32>;

@group(0) @binding(18)
var<storage,read_write> x_diff_2: array<f32>;

// 2D index to 1D index
fn zx(z: i32, x: i32) -> i32 {
    let index = x + z * infoI32.grid_size_x;

    return select(-1, index, x >= 0 && x < infoI32.grid_size_x && z >= 0 && z < infoI32.grid_size_z);
}

@compute
@workgroup_size(wsz, wsx)
fn derivative_z1(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    let dim: i32 = 0; // dimension (0 for Z, 1 for X)

    // Coefficients for finite difference method
    var coeff_diff = array<f32, 4>(1.19629, -0.07975, 0.00957, -0.00070);

    var derivative_result: f32 = 0.0;

    // Derivative in Z direction
    for (var i: i32 = 0; i < 4; i = i + 1) {
        let forward_idx = zx(z + i, x);
        let backward_idx = zx(z - i, x);
        if (forward_idx != -1 && backward_idx != -1) {
            derivative_result += coeff_diff[i] * p_present[forward_idx];
            derivative_result -= coeff_diff[i] * p_present[backward_idx];
        }
    }

    z_diff_1[zx(z, x)] = derivative_result;
}

@compute
@workgroup_size(wsz, wsx)
fn derivative_x1(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    // Coefficients for finite difference method
    var coeff_diff = array<f32, 4>(1.19629, -0.07975, 0.00957, -0.00070);

    var derivative_result: f32 = 0.0;

    // Derivative in X direction
    for (var i: i32 = 0; i < 4; i = i + 1) {
        let forward_idx = zx(z, x + i);
        let backward_idx = zx(z, x - i);
        if (forward_idx != -1 && backward_idx != -1) {
            derivative_result += coeff_diff[i] * p_present[forward_idx];
            derivative_result -= coeff_diff[i] * p_present[backward_idx];
        }
    }

    // boundary_factor == 1
    x_diff_1[zx(z, x)] = derivative_result;
}

@compute
@workgroup_size(wsz, wsx)
fn derivative_z2(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    let dim: i32 = 0; // dimension (0 for Z, 1 for X)

    // Coefficients for finite difference method
    var coeff_diff = array<f32, 4>(1.19629, -0.07975, 0.00957, -0.00070);

    var derivative_result: f32 = 0.0;

    // Derivative in Z direction
    for (var i: i32 = 0; i < 4; i = i + 1) {
        let forward_idx = zx(z + i, x);
        let backward_idx = zx(z - i, x);
        if (forward_idx != -1 && backward_idx != -1) {
            derivative_result += coeff_diff[i] * z_diff_1[forward_idx];
            derivative_result -= coeff_diff[i] * z_diff_1[backward_idx];
        }
    }

    // boundary_factor == 0
    if (z > 0 && z < infoI32.grid_size_z &&
        x > 0 && x < infoI32.grid_size_x) {
        z_diff_2[zx(z, x)] = derivative_result;
    }
}

@compute
@workgroup_size(wsz, wsx)
fn derivative_x2(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    let dim: i32 = 1; // dimension (0 for Z, 1 for X)

    // Coefficients for finite difference method
    var coeff_diff = array<f32, 4>(1.19629, -0.07975, 0.00957, -0.00070);

    var derivative_result: f32 = 0.0;

    // Derivative in X direction
    for (var i: i32 = 0; i < 4; i = i + 1) {
        let forward_idx = zx(z, x + i);
        let backward_idx = zx(z, x - i);
        if (forward_idx != -1 && backward_idx != -1) {
            derivative_result += coeff_diff[i] * x_diff_1[forward_idx];
            derivative_result -= coeff_diff[i] * x_diff_1[backward_idx];
        }
    }

    // boundary_factor == 0
    if (z > 0 && z < infoI32.grid_size_z &&
        x > 0 && x < infoI32.grid_size_x) {
        x_diff_2[zx(z, x)] = derivative_result;
    }
}

@compute
@workgroup_size(wsz, wsx)
fn sim(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    p_future[zx(z, x)] = cfl[zx(z, x)] * f32(z_diff_2[zx(z, x)] + x_diff_2[zx(z, x)]);

    p_future[zx(z, x)] += ((2. * p_present[zx(z, x)]) - p_past[zx(z, x)]);

    if (z == infoI32.source_z && x == infoI32.source_x)
    {
        p_future[zx(z, x)] += source[infoI32.i];
    }

    p_past[zx(z, x)] = p_present[zx(z, x)];
    p_present[zx(z, x)] = p_future[zx(z, x)];
}

@compute
@workgroup_size(1)
fn incr_time() {
    infoI32.i += 1;
}
