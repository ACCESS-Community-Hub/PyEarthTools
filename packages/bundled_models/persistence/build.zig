const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const lib_persistence_zig = b.addLibrary(.{
        .name = "persistence_zig",
        .linkage = .dynamic,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/lib/zig/lib.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    lib_persistence_zig.linkLibC();
    b.installArtifact(lib_persistence_zig);
}
