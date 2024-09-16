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

@group(0) @binding(19)
var<storage,read_write> shifted_u: array<f32>;

@group(0) @binding(20)
var<storage,read_write> dim: i32;

@group(0) @binding(21)
var<storage,read_write> boundary_factor: i32;

@group(0) @binding(22)
var<storage,read_write> is_z: i32;

@group(0) @binding(23)
var<storage,read_write> derivative_result: array<f32>;

// 2D index to 1D index
fn zx(z: i32, x: i32) -> i32 {
    let index = x + z * infoI32.grid_size_x;

    return select(-1, index, x >= 0 && x < infoI32.grid_size_x && z >= 0 && z < infoI32.grid_size_z);
}

// 2D index to 1D index
fn zx_2(z: i32, x: i32) -> i32 {
    let index = x + z * infoI32.grid_size_x;

    return select(-1, index, x >= 0 && x < infoI32.grid_size_x && z >= 0 && z < infoI32.grid_size_z + 1);
}

@compute
@workgroup_size(wsz, wsx)
fn update_z_diff_1(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    z_diff_1[zx(z, x)] = derivative_result[i32(zx_2(z, x) + 1)];
}

@compute
@workgroup_size(wsz, wsx)
fn update_x_diff_1_pt1(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    x_diff_1[zx(z, x)] = derivative_result[i32(zx_2(z, x) + 1)];
}

@compute
@workgroup_size(wsz, wsx)
fn update_x_diff_1_pt2(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    x_diff_1[zx(z, x)] = x_diff_1[zx(x, z)];
}

@compute
@workgroup_size(wsz, wsx)
fn update_z_diff_2(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    z_diff_2[zx(z, x)] = derivative_result[zx_2(z, x)];
}

@compute
@workgroup_size(wsz, wsx)
fn update_x_diff_2_pt1(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    x_diff_2[zx(z, x)] = derivative_result[zx_2(z, x)];
}

@compute
@workgroup_size(wsz, wsx)
fn update_x_diff_2_pt2(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    x_diff_2[zx(z, x)] = x_diff_2[zx(x, z)];
}

@compute
@workgroup_size(ws_derivative, wsx)
fn reset_derivative_result(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    derivative_result[zx_2(z, x)] = f32(0);
}

@compute
@workgroup_size(1)
fn update_for_dz1() {
    dim = 0;
    boundary_factor = 1;
    is_z = 1;
}

@compute
@workgroup_size(1)
fn update_for_dx1() {
    dim = 1;
    boundary_factor = 1;
    is_z = 0;
}

@compute
@workgroup_size(1)
fn update_for_dz2() {
    dim = 0;
    boundary_factor = 0;
    is_z = 1;
}

@compute
@workgroup_size(1)
fn update_for_dx2() {
    dim = 1;
    boundary_factor = 0;
    is_z = 0;
}

@compute
@workgroup_size(wsz, wsx)
fn set_shifted_u(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    if (boundary_factor == 1) {
        if (dim == 0) {
            shifted_u[zx(z, x)] = p_present[zx(z, x)];
        }
        else {
            shifted_u[zx(z, x)] = p_present[zx(x, z)];
        }
    }
    else {
        if (is_z == 1) {
            if (dim == 0) {
                shifted_u[zx(z, x)] = z_diff_1[zx(z, x)];
            }
            else {
                shifted_u[zx(z, x)] = z_diff_1[zx(x, z)];
            }
        }
        else {
            if (dim == 0) {
                shifted_u[zx(z, x)] = x_diff_1[zx(z, x)];
            }
            else {
                shifted_u[zx(z, x)] = x_diff_1[zx(x, z)];
            }
        }
    }
}

@compute
@workgroup_size(wsz, wsx)
fn derivative_z_1(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    var coeff_diff: array<f32, 4> = array<f32, 4>(1.19629, -0.07975, 0.00957, -0.00070);

    for (var i: i32 = 0; i < 4; i += 1) {
        if (i == 0) {
            derivative_result[zx_2(z, x)] += coeff_diff[i] * shifted_u[zx(z + i, x)];
            derivative_result[zx_2(z + i + 1, x)] -= coeff_diff[i] * shifted_u[zx(z, x)];
        }
        else if (i == 1) {
            if (z <= infoI32.grid_size_z - 2) {
                derivative_result[zx_2(z, x)] += coeff_diff[i] * shifted_u[zx(z + i, x)];
                derivative_result[zx_2(z + i + 1, x)] -= coeff_diff[i] * shifted_u[zx(z, x)];
            }
        }
        else if (i == 2) {
            if (z <= infoI32.grid_size_z - 3) {
                derivative_result[zx_2(z, x)] += coeff_diff[i] * shifted_u[zx(z + i, x)];
                derivative_result[zx_2(z + i + 1, x)] -= coeff_diff[i] * shifted_u[zx(z, x)];
            }
        }
        else if (i == 3) {
            if (z <= infoI32.grid_size_z - 4) {
                derivative_result[zx_2(z, x)] += coeff_diff[i] * shifted_u[zx(z + i, x)];
                derivative_result[zx_2(z + i + 1, x)] -= coeff_diff[i] * shifted_u[zx(z, x)];
            }
        }
    }
}

@compute
@workgroup_size(wsz, wsx)
fn derivative_x_1(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    var coeff_diff: array<f32, 4> = array<f32, 4>(1.19629, -0.07975, 0.00957, -0.00070);

    for (var i: i32 = 0; i < 4; i += 1) {
        if (i == 0) {
            derivative_result[zx_2(z, x)] += coeff_diff[i] * shifted_u[zx(z + i, x)];
            derivative_result[zx_2(z + i + 1, x)] -= coeff_diff[i] * shifted_u[zx(z, x)];
        }
        else if (i == 1) {
            if (z <= infoI32.grid_size_z - 2) {
                derivative_result[zx_2(z, x)] += coeff_diff[i] * shifted_u[zx(z + i, x)];
                derivative_result[zx_2(z + i + 1, x)] -= coeff_diff[i] * shifted_u[zx(z, x)];
            }
        }
        else if (i == 2) {
            if (z <= infoI32.grid_size_z - 3) {
                derivative_result[zx_2(z, x)] += coeff_diff[i] * shifted_u[zx(z + i, x)];
                derivative_result[zx_2(z + i + 1, x)] -= coeff_diff[i] * shifted_u[zx(z, x)];
            }
        }
        else if (i == 3) {
            if (z <= infoI32.grid_size_z - 4) {
                derivative_result[zx_2(z, x)] += coeff_diff[i] * shifted_u[zx(z + i, x)];
                derivative_result[zx_2(z + i + 1, x)] -= coeff_diff[i] * shifted_u[zx(z, x)];
            }
        }
    }
}

@compute
@workgroup_size(wsz, wsx)
fn derivative_z_2(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    var coeff_diff: array<f32, 4> = array<f32, 4>(1.19629, -0.07975, 0.00957, -0.00070);

    for (var i: i32 = 0; i < 4; i += 1) {
        if (i == 0) {
            derivative_result[zx_2(z, x)] += coeff_diff[i] * shifted_u[zx(z + i, x)];
            derivative_result[zx_2(z + i + 1, x)] -= coeff_diff[i] * shifted_u[zx(z, x)];
        }
        else if (i == 1) {
            if (z <= infoI32.grid_size_z - 2) {
                derivative_result[zx_2(z, x)] += coeff_diff[i] * shifted_u[zx(z + i, x)];
                derivative_result[zx_2(z + i + 1, x)] -= coeff_diff[i] * shifted_u[zx(z, x)];
            }
        }
        else if (i == 2) {
            if (z <= infoI32.grid_size_z - 3) {
                derivative_result[zx_2(z, x)] += coeff_diff[i] * shifted_u[zx(z + i, x)];
                derivative_result[zx_2(z + i + 1, x)] -= coeff_diff[i] * shifted_u[zx(z, x)];
            }
        }
        else if (i == 3) {
            if (z <= infoI32.grid_size_z - 4) {
                derivative_result[zx_2(z, x)] += coeff_diff[i] * shifted_u[zx(z + i, x)];
                derivative_result[zx_2(z + i + 1, x)] -= coeff_diff[i] * shifted_u[zx(z, x)];
            }
        }
    }
}

@compute
@workgroup_size(wsz, wsx)
fn derivative_x_2(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    var coeff_diff: array<f32, 4> = array<f32, 4>(1.19629, -0.07975, 0.00957, -0.00070);

    for (var i: i32 = 0; i < 4; i += 1) {
        if (i == 0) {
            derivative_result[zx_2(z, x)] += coeff_diff[i] * shifted_u[zx(z + i, x)];
            derivative_result[zx_2(z + i + 1, x)] -= coeff_diff[i] * shifted_u[zx(z, x)];
        }
        else if (i == 1) {
            if (z <= infoI32.grid_size_z - 2) {
                derivative_result[zx_2(z, x)] += coeff_diff[i] * shifted_u[zx(z + i, x)];
                derivative_result[zx_2(z + i + 1, x)] -= coeff_diff[i] * shifted_u[zx(z, x)];
            }
        }
        else if (i == 2) {
            if (z <= infoI32.grid_size_z - 3) {
                derivative_result[zx_2(z, x)] += coeff_diff[i] * shifted_u[zx(z + i, x)];
                derivative_result[zx_2(z + i + 1, x)] -= coeff_diff[i] * shifted_u[zx(z, x)];
            }
        }
        else if (i == 3) {
            if (z <= infoI32.grid_size_z - 4) {
                derivative_result[zx_2(z, x)] += coeff_diff[i] * shifted_u[zx(z + i, x)];
                derivative_result[zx_2(z + i + 1, x)] -= coeff_diff[i] * shifted_u[zx(z, x)];
            }
        }
    }
}

@compute
@workgroup_size(wsz, wsx)
fn sim(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    p_future[zx(z, x)] = cfl[zx(z, x)] * f32(f32(z_diff_2[zx(z, x)]) + f32(x_diff_2[zx(z, x)]));

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
