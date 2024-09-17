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

// 2D index to 1D index
fn zx(z: i32, x: i32) -> i32 {
    let index = x + z * infoI32.grid_size_x;

    return select(-1, index, x >= 0 && x < infoI32.grid_size_x && z >= 0 && z < infoI32.grid_size_z);
}

@compute
@workgroup_size(wsz, wsx)
fn first_derivatives(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    var pz: f32 = 0.;
    var px: f32 = 0.;

    if (z >= 2 && z <= infoI32.grid_size_z - 3) {
        pz = ((-1./12.) * p_present[zx(z + 2, x)] + (8./12.) * p_present[zx(z + 1, x)] - (8./12.) * p_present[zx(z - 1, x)] + (1./12.) * p_present[zx(z - 2, x)]) / (infoF32.dz);
    }
    if (x >= 2 && x <= infoI32.grid_size_x - 3) {
        px = ((-1./12.) * p_present[zx(z, x + 2)] + (8./12.) * p_present[zx(z, x + 1)] - (8./12.) * p_present[zx(z, x - 1)] + (1./12.) * p_present[zx(z, x - 2)]) / (infoF32.dx);
    }

    z_diff_1[zx(z, x)] = pz;
    x_diff_1[zx(z, x)] = px;
}

@compute
@workgroup_size(wsz, wsx)
fn second_derivatives_2(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    var pz: f32 = 0.;
    var px: f32 = 0.;

    if (z >= 2 && z <= infoI32.grid_size_z - 3) {
        pz = ((-1./12.) * z_diff_1[zx(z + 2, x)] + (8./12.) * z_diff_1[zx(z + 1, x)] - (8./12.) * z_diff_1[zx(z - 1, x)] + (1./12.) * z_diff_1[zx(z - 2, x)]) / (infoF32.dz);
    }
    if (x >= 2 && x <= infoI32.grid_size_x - 3) {
        px = ((-1./12.) * x_diff_1[zx(z, x + 2)] + (8./12.) * x_diff_1[zx(z, x + 1)] - (8./12.) * x_diff_1[zx(z, x - 1)] + (1./12.) * x_diff_1[zx(z, x - 2)]) / (infoF32.dx);
    }

    z_diff_2[zx(z, x)] = pz;
    x_diff_2[zx(z, x)] = px;
}

@compute
@workgroup_size(wsz, wsx)
fn second_derivatives(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    var pzz: f32 = 0.;
    var pxx: f32 = 0.;

    if (z >= 2 && z <= infoI32.grid_size_z - 3)
    {
        pzz = ((-1./12.) * p_present[zx(z + 2, x)] + (4./3.) * p_present[zx(z + 1, x)] - (5./2.) * p_present[zx(z, x)] + (4./3.) * p_present[zx(z - 1, x)] - (1./12.) * p_present[zx(z - 2, x)]) / (infoF32.dz * infoF32.dz);
    }
    if (x >= 2 && x <= infoI32.grid_size_x - 3)
    {
        pxx = ((-1./12.) * p_present[zx(z, x + 2)] + (4./3.) * p_present[zx(z, x + 1)] - (5./2.) * p_present[zx(z, x)] + (4./3.) * p_present[zx(z, x - 1)] - (1./12.) * p_present[zx(z, x - 2)]) / (infoF32.dx * infoF32.dx);
    }

    z_diff_2[zx(z, x)] = pzz;
    x_diff_2[zx(z, x)] = pxx;
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
