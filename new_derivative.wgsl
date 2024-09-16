// WGSL Shader for derivative calculation using 1D arrays

@group(0) @binding(0) var<storage, read> u: array<f32>; // Input 1D array representing 2D grid
@group(0) @binding(1) var<storage, read_write> result: array<f32>; // Output 1D array

// Grid information
struct GridInfo {
    grid_size_x: i32,
    grid_size_z: i32,
};

@group(0) @binding(2)
var<uniform> infoI32: GridInfo;


// 2D index to 1D index conversion function
fn zx(z: i32, x: i32) -> i32 {
    let index = x + z * infoI32.grid_size_x;

    return select(-1, index, x >= 0 && x < infoI32.grid_size_x && z >= 0 && z < infoI32.grid_size_z);
}

@compute @workgroup_size(8, 8) // Workgroup size (8x8 grid per workgroup)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let z: i32 = i32(global_id.x);
    let x: i32 = i32(global_id.y);
    let dim: u32 = global_id.z; // dimension (0 for Z, 1 for X)

    // Avoid out-of-bounds access
    if (z >= infoI32.grid_size_z || x >= infoI32.grid_size_x) {
        return;
    }

    var derivative_result: f32 = 0.0;

    if (dim == 0u) {
        // Derivative in Z direction
        for (var i = 0u; i < 4u; i = i + 1u) {
            let forward_idx = zx(z + i, x);
            let backward_idx = zx(z - i, x);
            if (forward_idx != -1 && backward_idx != -1) {
                derivative_result += coeff_diff[i] * u[forward_idx];
                derivative_result -= coeff_diff[i] * u[backward_idx];
            }
        }
    } else if (dim == 1u) {
        // Derivative in X direction
        for (var i = 0u; i < 4u; i = i + 1u) {
            let forward_idx = zx(z, x + i);
            let backward_idx = zx(z, x - i);
            if (forward_idx != -1 && backward_idx != -1) {
                derivative_result += coeff_diff[i] * u[forward_idx];
                derivative_result -= coeff_diff[i] * u[backward_idx];
            }
        }
    }

    // Apply the boundary factor logic
    if (boundary_factor == 1u) {
        result[zx(z, x)] = derivative_result;
    } else {
        // Handle boundary cases (you can adjust this further if needed)
        if (z > boundary_factor && z < infoI32.grid_size_z - boundary_factor &&
            x > boundary_factor && x < infoI32.grid_size_x - boundary_factor) {
            result[zx(z, x)] = derivative_result;
        }
    }
}
