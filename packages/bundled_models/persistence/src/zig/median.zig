const std = @import("std");
const nanf32 = std.math.nan(f32);

// ----------------------------------------------------------------------------
// Description:
//     Calculate median of three of an n-d array. Memory is allocated by numpy
//     (python) and passed in.
// ----------------------------------------------------------------------------
// Args:
//     idx_time: time index
//     shape: shape of input array
//     len_shape: shape of input array
//     arr_in: pointer to n-dimensional array
//     len_in: length of input array
//     arr_out: pointer to n-dimensional pre-allocated output
//     len_out: length of output array
// ----------------------------------------------------------------------------
fn medianofthree_numpy(
    idx_time: i32,
    shape: [*]i32,
    len_shape: i32,
    arr_in: [*]f32,
    len_in: i32,
    arr_out: [*]f32,
    len_out: i32,
) void {
    // UNIMPLEMENTED
    _ = &idx_time;
    _ = &shape;
    _ = &len_shape;
    _ = &arr_in;
    _ = &len_in;
    _ = &arr_out;
    _ = &len_out;
}

// ----------------------------------------------------------------------------
// Description:
//     Calculate median of three of scalars.
//     TODO: there may be a more efficient way.
// ----------------------------------------------------------------------------
// Alg:
//     input: (f32, f32, f32)
//     output: f32
//
//     {function state}
//     state:
//         - array[3]: container for valid inputs
//         - count: number of valid inputs (non-nan)
//
//     {nan filtering}
//     traverse inputs:
//         input is nan => skip
//         else => store in array and increment
//
//     {switch statement - NOTE: can be comptime}
//     compute median:
//         valid count = 0 => return NaN
//         valid count = 1 => return x[0]
//         valid count = 2 => return (x[0] + x[1]) / 2
//         valid count = 3 => return max(min(x[0], x[1]), x[2]) or similar
// ----------------------------------------------------------------------------
// Args:
//     x1, x2, x3: values to compute the median against
// ----------------------------------------------------------------------------
fn medianofthree_scalar_nanfiltered(x1: f32, x2: f32, x3: f32) f32 {
    var valid = [3]f32{ nanf32, nanf32, nanf32 };
    const xs = [3]f32{ x1, x2, x3 };
    var num_valid: u4 = 0;
    for (xs) |x| {
        if (!std.math.isNan(x)) {
            valid[num_valid] = x;
            num_valid += 1;
        }
    }

    std.debug.print("x1={},x2={},x3={},num_valid={}\n", .{ x1, x2, x3, num_valid });

    return medianofthree_scalar(num_valid, valid);
}

// ----------------------------------------------------------------------------
// Description:
//     Calculate median of a 3 element array, nans are masked. Unless the array
//     is all-nan in which case nan is returned. The switch prongs are comptime
//     resolvable since the choices are limited. Hopefully that makes it fast.
// ----------------------------------------------------------------------------
// Alg:
//     given [3]f32 array, x0, x1, x2 being the elements we need to compute the
//     median:
//
//     1. choosing x0' = min(x0, x1), x1' = min(x1, x2), x2' = min(x0, x2),
//        - {x0', x1', x2'} is guarenteed to have exactly two unique variables
//
//          (NOTE: variables, NOT values e.g. {x0, x1, x0} has two unique variables,
//                 {x0, x1, x2} and {x0, x0, x0} do not.)
//
//        - therefore, one of them must be the median.
//
//     2. the median has to be greater than the minimum of x1, x2, x3 so the
//        only guarenteed choice is to take the max of all three min-pairs:
//
//           median = max(max(x1', x2'), x0')
//
//     3. the expanded formula is given as:
//
//           max(max(min(x1, x2), min(x0, x2)), min(x0, x1))
//
//     4. note that max(min(x1, x2), min(x0, x2)):
//
//           if x2 < x1, x0 => x2
//           if x1 < x2 < x0 (or x0 < x2 < x1) => x2
//           if x0 < x1 < x2 (or x1 < x1 < x2) =>  max(x0, x1)
//
//        which is equivilent to:
//
//            min(max(x0, x1), x2)
//
//        i.e. I only choose x0 or x1 if x2 is an upper bound of {x0, x1}
//
//     5. substituing 4. into 3. we can now contract the number of operations
//        from 5 binary operations to 4. (though the compiler likely may have
//        done this anyway.)
//
//           median = max(min(max(x0, x1), x2), min(x0, x1))
// ----------------------------------------------------------------------------
// NOTE: the above describe the scenario where x0, x1, x2 are unique, without
// loss of generality. Duplicate entries do not change the outcome.
// ----------------------------------------------------------------------------
// Args:
//     num_valid: valid count to determine which operation to use for median
//     valid: the state array containing valid values
// ----------------------------------------------------------------------------
fn medianofthree_scalar(num_valid: u4, valid: [3]f32) f32 {
    return switch (num_valid) {
        0 => nanf32,
        1 => valid[0],
        2 => @as(f32, 0.5) * (valid[0] + valid[1]),
        3 => blk: {
            const x0: f32, const x1: f32, const x2: f32 = valid;
            const median = @max(@max(@min(x0, x1), @min(x1, x2)), @min(x0, x2));
            break :blk median;
        },
        else => nanf32,
    };
}

test "median of three test fleet" {
    // 0. median of all nan
    var x1: f32 = nanf32;
    var x2: f32 = nanf32;
    var x3: f32 = nanf32;
    var expect: f32 = nanf32;
    var result = medianofthree_scalar_nanfiltered(x1, x2, x3);
    try std.testing.expectEqual(std.math.isNan(expect), std.math.isNan(result));

    // 1. median of one
    x1 = nanf32;
    x2 = nanf32;
    x3 = 0.5;
    expect = 0.5;
    result = medianofthree_scalar_nanfiltered(x1, x2, x3);
    try std.testing.expectEqual(expect, result);

    // 2. median of two (mean)
    x1 = 5.0;
    x2 = nanf32;
    x3 = -10.0;
    expect = -2.5;
    result = medianofthree_scalar_nanfiltered(x1, x2, x3);
    try std.testing.expectEqual(expect, result);

    // 3. median of three (actually median)
    x1 = -5.0;
    x2 = 20.0;
    x3 = -10.0;
    expect = -5.0;
    result = medianofthree_scalar_nanfiltered(x1, x2, x3);
    try std.testing.expectEqual(expect, result);
}
