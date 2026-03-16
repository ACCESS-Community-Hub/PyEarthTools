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
pub fn medianofthree_split_nd(
    idx_time: i32,
    shape: [*]i32,
    len_shape: i32,
    arr_in: [*]f32,
    len_in: i32,
    arr_out: [*]f32,
    len_out: i32,
) void {
    // --- probably not optimal - for simplicity ---
    // var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    // defer arena.deinit();
    // const allocator = arena.allocator();
    // ---
    const shape_arr: []i32 = shape[0..@as(usize, @intCast(len_shape))];
    const len_chunk: usize, const len_outer: usize = blk: {
        var _prod_inner: usize = 1;
        var _prod_outer: usize = 1;
        for (shape_arr, 0..) |s, i| {
            const s_usize: usize = @intCast(s);
            if (i > idx_time) _prod_inner *= s_usize;
            if (i < idx_time) _prod_outer *= s_usize;
        }
        break :blk .{ _prod_inner, _prod_outer };
    };

    // safety
    std.debug.assert(@as(usize, @intCast(len_out)) == len_chunk * len_outer);
    std.debug.assert(@as(usize, @intCast(len_in)) == shape[@as(usize, @intCast(idx_time))] * len_out);

    for (0..len_outer) |i| {
        // ---
        // start
        const chunk_idxs = len_chunk * i;
        // --- 3 equal length chunks representing time indices ---
        // TODO: a more generic strategy required for historically lengthier metrics
        const chunk_idx1 = 3 * chunk_idxs;
        const chunk_idx2 = chunk_idx1 + len_chunk;
        const chunk_idx3 = chunk_idx2 + len_chunk;
        // ---
        // end
        const chunk_idxe = chunk_idx3 + len_chunk;
        // ---

        // get chunks that are contiguous, in one go to avoid jumps
        // slice view of contiguous cuhnks, so memory allocation not required.
        const cntg_chunk1 = arr_in[chunk_idx1..chunk_idx2];
        const cntg_chunk2 = arr_in[chunk_idx2..chunk_idx3];
        const cntg_chunk3 = arr_in[chunk_idx3..chunk_idxe];

        // fill output array
        for (0..len_chunk) |j| {
            arr_out[chunk_idxs + j] = medianofthree_scalar_nanfiltered(
                cntg_chunk1[j],
                cntg_chunk2[j],
                cntg_chunk3[j],
            );
        }
    }
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
pub fn medianofthree_scalar_nanfiltered(x1: f32, x2: f32, x3: f32) f32 {
    var valid = [3]f32{ nanf32, nanf32, nanf32 };
    const xs = [3]f32{ x1, x2, x3 };
    var num_valid: u4 = 0;
    for (xs) |x| {
        if (!std.math.isNan(x)) {
            valid[num_valid] = x;
            num_valid += 1;
        }
    }

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
            const median = @max(@min(@max(x0, x1), x2), @min(x0, x1));
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

test "median of three nd" {
    {
        var test_arr_in: [5][4][3][6][3]f32 = undefined;
        var test_arr_out: [5][4][1][6][3]f32 = undefined;
        const total_len = 5 * 4 * 3 * 6 * 3;
        for (0..total_len) |i| {
            const arr_ptr: [*]f32 = @ptrCast(&test_arr_in);
            arr_ptr[i] = @as(f32, @floatFromInt(i));
        }
        var shape = [_]i32{ 5, 4, 3, 6, 3 };
        medianofthree_split_nd(
            2,
            &shape,
            5,
            @ptrCast(&test_arr_in),
            @ptrCast(&test_arr_out),
            total_len / 3,
        );
        const arr_out_ptr: [*]f32 = @ptrCast(&test_arr_out);
        const arr_in_ptr: [*]f32 = @ptrCast(&test_arr_in);
        std.debug.print("{any}\n", .{arr_in_ptr[0..total_len]});
        std.debug.print("{any}\n", .{arr_out_ptr[0..(total_len / 3)]});
    }

    {
        var test_arr_in = [2][3]f32{
            [_]f32{ 2, -5, 4 },
            [_]f32{ 5, 100, -2 },
        };
        var test_arr_out: [2][1]f32 = undefined;
        var shape = [_]i32{ 2, 3 };
        const out = medianofthree_scalar_nanfiltered(2, -5, 4);
        medianofthree_split_nd(
            1,
            &shape,
            2,
            @ptrCast(&test_arr_in),
            @ptrCast(&test_arr_out),
            2,
        );
        std.debug.print("\n{any}\n", .{out});
        std.debug.print("\n{any}\n", .{test_arr_in});
        std.debug.print("\n{any}\n", .{test_arr_out});
    }
}
