#!/usr/bin/env bash

# --- zig-test (worker-only) ---
# description:
#   zig unit tests to check the state transition flow for the zig
#   server/worker
# requirements:
#   - entr for file watch
#   - pthread (usually included in system libs if on linux)
#   - czmq (high level apis)
#   - zmq and any dependencies of czmq
#   - TODO: link netcdf lib for file loading
# ---
ls worker.zig | entr bash -c 'clear && zig test -lc -lpthread -lczmq -lzmq worker.zig'
# ---

# --- python-test (manager-only) ---
# description:
#   python unit tests to check the state transition flow for the python
#   client/manager
# requirements:
#   - whatever is on PET
#   - pyzmq (or similar bindings)
# ---
echo "TODO: python client tests not implemented"
# ---

# --- end-to-end ----
# The following is a implementation goal, NOT current state.
#
# description:
#   This test will perform an end to end processing of netcdf file using zmq.
#
#   The default pattern used is intentionally contrived - for learning purposes.
#   There are two planes of communication, control and data.
#
#   The control plane is for metadata flows, and is BIDIRECTIONAL between the
#   worker and client. It's tcp based (though IPC can work)
#
#   The data plane has two UNIDIRECTIONAL connections, one receiving
#   information from the controller (inproc - fast). The other sending
#   information to the client directly (ipc - not as fast, but better than tcp)
#
#   Here the controller in the worker run in the zig process is essentially a
#   proxy that listens to client (python commands or function calls) and
#   offloads them to a task. The task(s) then crunch the numbers and spit it
#   back to a "SINK" socket that the python process can listen to in order to
#   consolidate the results.
#
#   From the perspective of someone running e.g.
#        `call "median_of_three" with chunks x y z from file F.nc`
#        (note this is a made up message protocol)
#
#   is effectively the same as running median_of_three(file, [x, y, z]) as a
#   python function. In fact, the interface will still likely be the same,
#   except for python's ability to dynamically translate symbols to strings
#   making it simple to serialize this information into a zeromq socket.
#
#   Process:
#   1. Python initializes the zig executable (worker) and connects to it.
#   2. Zig worker responds with information.
#
#   Simplifications can definitely be made down the track including the usage of:
#   - zpoller | we use our own loops/sleep
#   - zactors | we have actions through dedicated socket)
#   - zproxy  | we are essentially having two state machines in zig, 1 acting
#             | as the controller and the other for work performed by each task
#   - zloop (event based)
#   - ... and many more convenience features
#
#   NOTE: The equivilent function is median_of_three in persistence models.
#
#   (Linux-only, zig can have num_workers > 1, in conjunction with pthreads to
#   allow for lean work to be done, to bypass IO bound limitations).
# requirements:
#   - ditto above
echo "TODO: end-to-end tests not implemented"

# ---
