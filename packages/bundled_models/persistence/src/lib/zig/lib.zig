const std = @import("std");
const median = @import("./median.zig");

export fn median_of_three(x1: f32, x2: f32, x3: f32) f32 {
    return median.medianofthree_scalar_nanfiltered(x1, x2, x3);
}
