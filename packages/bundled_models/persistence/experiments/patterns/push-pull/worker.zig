// ============================================================================
// TODO: [ ] ClientAction - these needs to be caught by the controller
// TODO: [ ] WorkerTask pthreads(inproc)
// ----------------------------------------------------------------------------
// NOTE: ---------------------
// NOTE: @ = bind; > = connect
// NOTE: ---------------------
// NOTE: we typically want to:
// NOTE:   - bind(@): if we want to LISTEN for information from an
// NOTE:     external party.
// NOTE:   - connect(>): if we want to REQUEST for information from an
// NOTE:     external party.
// NOTE: therefore for our ventilator:
// NOTE:   a) > => PULL data from external party (party=client)
// NOTE:   b) @ => PUSH response to external party (party=server)
// ============================================================================

const std = @import("std");
const czmq = @cImport({
    // See https://github.com/ziglang/zig/issues/515
    @cDefine("_NO_CRT_STDIO_INLINE", "1");
    @cInclude("stdio.h");
    @cInclude("czmq.h");
});

const WorkerError = error{
    TaskAlreadyExists,
    TaskDoesNotExist,
    ClientActionInvalid,
    NotImplemented,
};

// === PUSH ===
// Task header when returning task id, used for new task.
const C_TASK_ID_KEY: *const u8 = "C_TASK_ID";
// Task with id not found in controller hashmap.
const C_TASK_NOT_FOUND: *const u8 = "C_TASK_NOT_FOUND";
// Check whether client is ready.
const C_CLIENT_READY: *const u8 = "C_CLIENT_READY";

// PUSH: single "ready state" - controller
const ControllerState = enum {
    controller_ok,
    controller_busy,
    controller_error,
    controller_stopped,

    fn to_str(self: ControllerState) [*c]const u8 {
        switch (self) {
            .controller_ok => return "C_OK",
            .controller_busy => return "C_BUSY",
            .controller_error => return "C_ERROR",
            .controller_stopped => return "C_STOPPED",
        }
    }
};

// ---------------------------------------------------------------------------
// PUSH: task state to send back to client
//
// NOTE:
//   - There is a hidden state here where if the task no longer exists it
//     may have been cleaned up
//   - The catch-22 is when a task is cleaned up, it can't be queried, in
//     such a circumstance, the controller will just return something akin to
//     "TASK NOT FOUND". Depending on the client state, this can mean:
//       - "ERROR", OR
//       - more likely, "TASK HAS BEEN SUCCESSFULLY CLEANED UP"
// ---------------------------------------------------------------------------
const TaskState = enum {
    task_ready,
    task_working,
    task_finished,
    task_error,

    fn to_str(self: TaskState) [*c]const u8 {
        switch (self) {
            .task_ready => return "T_READY",
            .task_working => return "T_WORKING",
            .task_finished => return "T_FINISHED",
            .task_error => return "T_ERROR",
        }
    }
};

// === PULL ===
const ClientAction = enum {
    controller_state,
    controller_end,
    task_state,
    task_create,
    task_end,

    fn init_from_str(s: [*]u8) !ClientAction {
        if (czmq.strcmp(s, ClientAction.controller_state.to_str()) == 0)
            return ClientAction.controller_state;
        if (czmq.strcmp(s, ClientAction.controller_end.to_str()) == 0)
            return ClientAction.controller_end;
        if (czmq.strcmp(s, ClientAction.task_state.to_str()) == 0)
            return ClientAction.task_state;
        if (czmq.strcmp(s, ClientAction.task_end.to_str()) == 0)
            return ClientAction.task_end;
        if (czmq.strcmp(s, ClientAction.task_create.to_str()) == 0)
            return ClientAction.task_end;
        return WorkerError.ClientActionInvalid;
    }

    fn to_str(self: ClientAction) [*c]const u8 {
        switch (self) {
            .controller_state => return "A_CONTROLLER_STATE",
            .controller_end => return "A_CONTROLLER_END",
            .task_state => return "A_TASK_STATE",
            .task_create => return "A_TASK_CREATE",
            .task_end => return "A_TASK_DISCONNECT",
        }
    }
};

const ControlSocket = struct {
    // --- members ---
    // recv action from client (includes "GET_STATE")
    ptr_sock_pull: ?*czmq.zsock_t,
    // send state to client
    ptr_sock_push: ?*czmq.zsock_t,

    // --- const ---
    // socket to pull actions
    const PULL_PORT: u32 = 5558;
    // socket to push replies
    const PUSH_PORT: u32 = 5559;

    // -----------------------------------------------------------------------
    // Initialize the Socket with the pull socket (bind).
    // The push (connect) will be established after initial comms from the
    // client.
    // -----------------------------------------------------------------------
    fn init() ControlSocket {
        std.debug.print("Setting up pull socket connection tcp://:{d}\n", .{PULL_PORT});
        const uri_pull = std.fmt.comptimePrint("@tcp://127.0.0.1:{d}", .{PULL_PORT});
        const ptr_sock_pull = czmq.zsock_new_pull(uri_pull);
        czmq.zclock_sleep(10);
        std.debug.assert(ptr_sock_pull != null);

        // --- builder ---
        return ControlSocket{
            .ptr_sock_push = null,
            .ptr_sock_pull = ptr_sock_pull,
        };
    }

    // -----------------------------------------------------------------------
    // Initializing the push connection is driven by the controller (unlike
    // the pull connection which should always exist if the controller
    // exists).
    // -----------------------------------------------------------------------
    fn connect_push_sock(self: *ControlSocket) void {
        if (self.ptr_sock_push) |_| return;

        // --- push ---
        std.debug.print("Setting up push socket connection tcp://:{d}\n", .{PUSH_PORT});
        const uri_push = std.fmt.comptimePrint(">tcp://127.0.0.1:{d}", .{PUSH_PORT});
        const ptr_sock_push: ?*czmq.zsock_t = czmq.zsock_new_push(uri_push);
        czmq.zclock_sleep(10);

        // update the push socket
        self.ptr_sock_push = ptr_sock_push;
    }

    fn destroy_push(self: *ControlSocket) void {
        czmq.zsock_destroy(&self.ptr_sock_push);
    }

    fn destroy_pull(self: *ControlSocket) void {
        czmq.zsock_destroy(&self.ptr_sock_pull);
    }

    fn deinit(self: *ControlSocket) void {
        self.destroy_push();
        self.destroy_pull();
    }
};

// TODO: PLACEHOLDER
const DataSocket = opaque {};

// --------------------------------------------------------------------------
// This implementation is to better understand the control flow of zmq.
//
// The controller is _actually_ a proxy for the upstream client which maybe
// implemented in a different language.
//
// It's purpose is to parse the messages and map an action to a "task" as well
// as return any feedback control messages upstream.
// Briefly the controller:
//     1. handles the control socket state [LIMIT=1, push-pull pair]
//     2. handles the task management [LIMIT=2 tasks]
//        above mentioned LIMITs are to be lifted when this framework is
//        evaluated, tested and is no longer experimental.
//
// run_stm() Runs the main control state machine. This:
//     - checks the control socket for incoming requests
//     - updates the controller or managed task states accordingly
//     - currently attempts to receive in a loop with a hard-coded sleep
//       (lazy pirate)
//     - recv's messages in bulk up to a limit of 100
//
// it is HIGHLY likely that a proxy or router/dealer combo does exactly the
// same thing and should be explored.
//
// ---
// controller-transition:
//     C_OK      -> task_count == MAX_COUNT
//               -> C_BUSY
//
//     C_BUSY    -> task_count < MAX_COUNT
//               -> C_OK
//
//     ANY       -> error encountered
//               -> clean and destroy
//
// ---
// pull-push:
//     pull A_CONTROLLER_STATE.c_id
//          => push C_STATE.c_id=<state>
//
//     pull A_TASK_STATE.c_id.t_id
//          => push T_STATE.c_id.t_id=<state>
//
//     pull A_TASK_CREATE.c_id.t_id
//          => push T_STATE.c_id.t_id=<state> (state=READY)
//
//     pull A_TASK_DISCONNECT.c_id.t_id
//          => delete task
//
// (ids above are for sanity checks only)
//
// ---
// client-transition (recommendation only):
//
//     [1] check if controller is ready :
//             loop(push A_CONTROLLER_STATE)
//                  C_OK           => request tasks until max burst count
//                                    (agreed upon limit)
//                  C_BUSY         => wait => goto [1]
//                  timeout        => exit
//                  C_ERROR        => exit
//
//     [2] task request:
//             task_create, loop(push A_TASK_STATE):
//                 C_OK && T_READY => goto [3]
//                 C_BUSY          => wait => goto [2]
//                 timeout         => exit
//                 *_ERROR         => exit
//
//     [3] send multiframe based on "ACTION spec" (see: Task)
//             loop(push A_TASK_STATE)
//                 T_WORKING       => goto [4]
//                 timeout         => exit
//                 *_ERROR         => exit
//
//     [4] clean up or kill task
//             loop(push A_TASK_STATE):
//                 T_FINISHED      => push A_TASK_DISCONNECT
//                 timeout         => exit
//                 *_ERROR         => exit
//
// FUTUREWORK: use zpoller() instead.
// ---------------------------------------------------------------------------
const Controller = struct {
    // allocation
    a: *std.mem.Allocator,
    // controller id
    id_c: usize,
    // running task id - auto incremented
    id_t_next: u32,
    // task states
    tasks: std.AutoHashMap(usize, Task),
    // controller state
    state_c: ControllerState,
    // used with TaskState to set the socket state for each task
    sock_c: ControlSocket,

    const MAX_TASKS = 2;
    const MAX_BURST_CONTROL_MSGS = 100;

    fn init(alloc: *std.mem.Allocator, id_c: usize) Controller {
        return Controller{
            .a = alloc,
            .id_c = id_c,
            .id_t_next = 0,
            .tasks = std.AutoHashMap(usize, Task).init(alloc.*),
            .state_c = .controller_ok,
            .sock_c = ControlSocket.init(),
        };
    }

    fn deinit(self: *Controller) void {
        // de-initialize tasks
        var iter = self.tasks.valueIterator();
        while (iter.next()) |t| t.deinit();
        self.tasks.deinit();

        // remove connections and bindings socket
        self.sock_c.deinit();
    }

    fn run_stm(self: *Controller) !void {
        const cycles_ms = 10; // wait before reconnecting when forced to stop
        var cycles: u32 = 0; // unused because blocking, only useful when we are doing proper polling
        while (true) : (cycles += 1) {
            // blocking
            var maybe_msg = czmq.zstr_recv(self.sock_c.ptr_sock_pull);
            // received message => process
            if (maybe_msg) |msg| {
                const action = try ClientAction.init_from_str(msg);
                switch (action) {
                    // controller state was requested
                    .controller_state => {
                        // container state is being checked => wake up
                        cycles = 0;
                        std.debug.print("CLIENT(recv): check controller state.\n", .{});
                        // reconnect to socket if not still connected
                        self.sock_c.connect_push_sock();
                        if (self.sock_c.ptr_sock_push) |_| {
                            // non-blocking
                            _ = czmq.zstr_send(self.sock_c.ptr_sock_push, self.state_c.to_str());
                        }
                        czmq.zstr_free(&maybe_msg);
                    },
                    // client wants to stop the controller
                    .controller_end => {
                        std.debug.print("CLIENT(recv): stop controller.\n", .{});
                        self.state_c = .controller_stopped;
                        if (self.sock_c.ptr_sock_push) |_| {
                            // non-blocking
                            _ = czmq.zstr_send(self.sock_c.ptr_sock_push, self.state_c.to_str());
                        }
                        czmq.zstr_free(&maybe_msg);
                    },
                    else => {
                        return WorkerError.NotImplemented;
                    },
                }
            }
            czmq.zclock_sleep(cycles_ms);
        }
    }

    fn new_task(self: *Controller) !*Task {
        if (self.tasks.get(self.id_t_next)) |_| {
            return WorkerError.TaskAlreadyExists;
        } else {
            try self.tasks.put(self.id_t_next, Task.init(self.id_t_next));
            const ptr_t = self.tasks.getPtr(self.id_t_next).?;
            self.id_t_next += 1;
            return ptr_t;
        }
    }

    fn remove_task(self: *Controller, id_t: usize) !void {
        var t_ptr = self.tasks.getPtr(id_t);
        t_ptr.deinit();
        if (!self.tasks.remove(id_t)) {
            return WorkerError.TaskDoesNotExist;
        }
    }
};

// --------------------------------------------------------------------------
// Tasks currently flow from start to finish
// - retries and interrupts are NOT supported. Instead the client can trigger
//   a command to end the task and create a new one to retry. IF the task has
//   stalled, It's a DUAL responsiblity model:
//      - worker to raise an error (or log it).
//      - client to raise an error, timeout.
//      - but NO imposition is made on fault tolerance (unlike elixir/erlang).
// - If a task is multithreaded, it MAY be killed. Obviously this won't work
//   with a single process, since it'll also kill the controller.
//
// For now in-scope: single-threaded processing ONLY.
//
// ---
// task-transition:
//     T_READY   -> receive action
//               -> call function
//               -> T_WORKING
//
//     T_WORKING -> finished processing
//               -> T_FINISHED
//
//     ANY       -> error encountered
//               -> T_ERROR
//
//     ANY       -> end received
//               -> clean and destroy
//
// ---
// push-pull:
//     pull:
//         A_TASK_CREATE=<fn>           -- frame 0
//         A_TASK_CREATE=<arg1>=<val1>  -- frame 1
//         ...                          -- ...
//         A_TASK_CREATE=<argN>=<valN>  -- frame N
//
//     push:
//         R_TASK=<type>
//         R_TASK=<resultChunk1>        -- frame 0
//         R_TASK=<resultChunk2>        -- frame 1
//         ...                          -- ...
//         R_TASK=<resultChunkN>        -- frame N
//
// ---
// client-transition (recommendation only):
//     (assuming controller has polled finished task)
//     task state = T_FINISHED => pull multiframe result
//                             => parse
//                                if success, tell controller to end task
//                                if failure,
//                                    log error
//                                    ask controlelr to end task (inproc)
//                                    OR send retry (NOT SUPPORTED CURRENTLY)
// --------------------------------------------------------------------------
const Task = struct {
    // task id
    id_t: usize,
    socket: ?*DataSocket,
    state: TaskState,

    fn init(id_t: usize) Task {
        // create socket
        return Task{
            .id_t = id_t,
            .socket = null,
            .state = TaskState.task_ready,
        };
    }

    fn set_state(self: *Task, new_state: TaskState) void {
        self.state = new_state;
    }

    fn deinit(self: *Task) void {
        // TODO: remove task level socket connection and bindings
        _ = self;
    }
};

// This should run on a separate thread
fn __test_worker_controller_stm(a: *std.mem.Allocator) void {
    var c = Controller.init(a, 1);
    defer c.deinit();
    c.run_stm() catch {};
}

pub fn main() !void {
    // TODO: replace with actual workflow
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    var a = arena.allocator();
    // --- start worker ---
    // sorker will have its pull socket (@), but not push - until the first
    // status message is received.
    const t_worker = try std.Thread.spawn(
        .{ .allocator = a },
        __test_worker_controller_stm,
        .{&a},
    );
    t_worker.join();
}

test "task state" {
    var t = Task.init(10);
    std.debug.print("t.state = {s}\n", .{t.state.to_str()});
    t.set_state(.task_working);
    std.debug.print("t.state = {s}\n", .{t.state.to_str()});
}

test "controller tasks" {
    var a = std.testing.allocator;
    var c = Controller.init(&a, 0);
    defer c.deinit();
    var t1 = try c.new_task();
    t1.set_state(TaskState.task_finished);
    var t2 = try c.new_task();
    t2.set_state(TaskState.task_working);
    std.debug.print(
        "{d}\nt1: {any}\nt2: {any}\n",
        .{ c.tasks.count(), t1.*, t2.* },
    );
}

test "controller status check" {

    // --- start worker ---
    // sorker will have its pull socket (@), but not push - until the first
    // status message is received.
    const t_worker = try std.Thread.spawn(
        .{ .allocator = std.testing.allocator },
        __test_worker_controller_stm,
        .{},
    );

    // --- create sockets to mimic client ---
    // 1. push (worker pull) should already be available so connect to it (>)
    var sock_client_push: ?*czmq.zsock_t = czmq.zsock_new_push(">tcp://127.0.0.1:5558");
    // 2. bind pull socket (@), this is independent of the worker
    var sock_client_pull: ?*czmq.zsock_t = czmq.zsock_new_pull("@tcp://127.0.0.1:5559");
    // imposing wait times so test outputs are clearly seen.
    czmq.zclock_sleep(100);

    // --- client checks controller status ---
    // (this should prompt the worker to connect a push socket to reply)
    _ = czmq.zstr_send(sock_client_push, ClientAction.controller_state.to_str());
    std.debug.print("Client-sent: {s}\n", .{ClientAction.controller_state.to_str()});
    czmq.zclock_sleep(100);

    // --- worker reply ---
    // EXPECT: C_OK
    {
        const maybe_msg = czmq.zstr_recv(sock_client_pull);
        if (maybe_msg) |msg| std.debug.print("Client-received: {s}\n", .{msg});
        _ = czmq.free(maybe_msg);
    }
    czmq.zclock_sleep(100);

    // --- kill worker remotely ---
    // EXPECT: C_STOPPED
    _ = czmq.zstr_send(sock_client_push, ClientAction.controller_end.to_str());
    std.debug.print("Client-sent: {s}\n", .{ClientAction.controller_end.to_str()});
    czmq.zclock_sleep(100);

    // --- worker will reply first then stop ---
    // the worker is essentially in a held state until the state machine
    // restarts through some other mechanism, or is cleaned up.
    {
        const maybe_msg = czmq.zstr_recv(sock_client_pull);
        if (maybe_msg) |msg| std.debug.print("Client received: {s}\n", .{msg});
        _ = czmq.free(maybe_msg);
    }
    czmq.zclock_sleep(100);

    // --- cleanup client ---
    // print received status message and reply.
    // this will cleanup any sockets.
    t_worker.join();
    // destroy sockets
    _ = czmq.zsock_destroy(&sock_client_push);
    _ = czmq.zsock_destroy(&sock_client_pull);
}
