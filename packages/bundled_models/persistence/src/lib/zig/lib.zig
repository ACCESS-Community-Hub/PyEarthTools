const std = @import("std");
const median = @import("./median.zig");

export fn median_of_three(x1: f32, x2: f32, x3: f32) f32 {
    return median.medianofthree_scalar_nanfiltered(x1, x2, x3);
}

export fn median_of_three_nd(
    idx_time: i32,
    shape: [*]i32,
    len_shape: i32,
    arr_in: [*]f32,
    len_in: i32,
    arr_out: [*]f32,
    len_out: i32,
) void {
    median.medianofthree_split_nd(
        idx_time,
        shape,
        len_shape,
        arr_in,
        len_in,
        arr_out,
        len_out,
    );
}
