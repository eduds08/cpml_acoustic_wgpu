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

@group(0) @binding(6) // c
var<storage,read> c: array<f32>;

@group(0) @binding(7)
var<storage,read_write> z_diff_1: array<f32>;

@group(0) @binding(8)
var<storage,read_write> z_diff_2: array<f32>;

@group(0) @binding(9)
var<storage,read_write> x_diff_1: array<f32>;

@group(0) @binding(10)
var<storage,read_write> x_diff_2: array<f32>;

@group(0) @binding(11)
var<storage,read_write> phi_z: array<f32>;

@group(0) @binding(12)
var<storage,read_write> phi_x: array<f32>;

@group(0) @binding(13)
var<storage,read_write> absorption_z: array<f32>;

@group(0) @binding(14)
var<storage,read_write> absorption_x: array<f32>;

@group(0) @binding(15)
var<storage,read_write> psi_z: array<f32>;

@group(0) @binding(16)
var<storage,read_write> psi_x: array<f32>;

@group(0) @binding(17)
var<storage,read_write> is_z_absorption: array<i32>;

@group(0) @binding(18)
var<storage,read_write> is_x_absorption: array<i32>;

// 2D index to 1D index
fn zx(z: i32, x: i32) -> i32 {
    let index = x + z * infoI32.grid_size_x;

    return select(-1, index, x >= 0 && x < infoI32.grid_size_x && z >= 0 && z < infoI32.grid_size_z);
}

@compute
@workgroup_size(wsz, wsx)
fn forward_partial(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    var pz: f32 = 0.;
    var px: f32 = 0.;

    if (z + 1 < infoI32.grid_size_z) {
        pz = (p_present[zx(z + 1, x)] - p_present[zx(z, x)]) / infoF32.dz;
    }
    if (x + 1 < infoI32.grid_size_x) {
        px = (p_present[zx(z, x + 1)] - p_present[zx(z, x)]) / infoF32.dx;
    }

    z_diff_1[zx(z, x)] = pz;
    x_diff_1[zx(z, x)] = px;
}

@compute
@workgroup_size(wsz, wsx)
fn backward_partial(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    var pz: f32 = 0.;
    var px: f32 = 0.;

    if (z - 1 >= 0) {
        pz = (z_diff_1[zx(z, x)] - z_diff_1[zx(z - 1, x)]) / infoF32.dz;
    }
    if (x - 1 >= 0) {
        px = (x_diff_1[zx(z, x)] - x_diff_1[zx(z, x - 1)]) / infoF32.dx;
    }

    z_diff_2[zx(z, x)] = pz;
    x_diff_2[zx(z, x)] = px;
}

@compute
@workgroup_size(wsz, wsx)
fn after_d1(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    if (is_z_absorption[zx(z, x)] == 1) {
        phi_z[zx(z, x)] = absorption_z[zx(z, x)] * phi_z[zx(z, x)] + (absorption_z[zx(z, x)] - 1) * z_diff_1[zx(z, x)];
        z_diff_1[zx(z, x)] += phi_z[zx(z, x)];
    }
    if (is_x_absorption[zx(z, x)] == 1) {
        phi_x[zx(z, x)] = absorption_x[zx(z, x)] * phi_x[zx(z, x)] + (absorption_x[zx(z, x)] - 1) * x_diff_1[zx(z, x)];
        x_diff_1[zx(z, x)] += phi_x[zx(z, x)];
    }
}

@compute
@workgroup_size(wsz, wsx)
fn after_d2(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    if (is_z_absorption[zx(z, x)] == 1) {
        psi_z[zx(z, x)] = absorption_z[zx(z, x)] * psi_z[zx(z, x)] + (absorption_z[zx(z, x)] - 1) * z_diff_2[zx(z, x)];
        z_diff_2[zx(z, x)] += psi_z[zx(z, x)];
    }
    if (is_x_absorption[zx(z, x)] == 1) {
        psi_x[zx(z, x)] = absorption_x[zx(z, x)] * psi_x[zx(z, x)] + (absorption_x[zx(z, x)] - 1) * x_diff_2[zx(z, x)];
        x_diff_2[zx(z, x)] += psi_x[zx(z, x)];
    }
}

@compute
@workgroup_size(wsz, wsx)
fn sim(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    p_future[zx(z, x)] = (c[zx(z, x)] * c[zx(z, x)]) * (z_diff_2[zx(z, x)] + x_diff_2[zx(z, x)]) * (infoF32.dt * infoF32.dt);

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
