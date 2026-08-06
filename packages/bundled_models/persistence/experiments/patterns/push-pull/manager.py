if __name__ == "__main__":
    import zmq
    import time

    PUSH_URL = "tcp://127.0.0.1:5558"
    PULL_URL = "tcp://127.0.0.1:5559"
    # best to keep with strict context management in python impl
    with zmq.Context() as ctx:
        with ctx.socket(zmq.PUSH) as s_push, ctx.socket(zmq.PULL) as s_pull:
            print(">>> TEST get state (first time) with sleep 100ms")
            # first control message
            with s_push.connect(PUSH_URL):
                print("client: A_CONTROLLER_STATE")
                s_push.send_string("A_CONTROLLER_STATE")
            # ---> sleep <---
            time.sleep(0.1)
            # wait before receiving for initialization
            with s_pull.bind(PULL_URL):
                reply = s_pull.recv_string()
                print("worker: " + reply)
                assert reply == "C_OK"

            print(">>> TEST get state no sleep")
            # second control message wait
            with s_push.connect(PUSH_URL):
                print("client: A_CONTROLLER_STATE")
                s_push.send_string("A_CONTROLLER_STATE")
            # ---> no sleep <---
            # should receive the reply instantly
            with s_pull.bind(PULL_URL):
                reply = s_pull.recv_string()
                print("worker: " + reply)
                assert reply == "C_OK"

            print(">>> TEST kill worker")
            # stop worker (normally this is irrecoverable, unless the worker
            # auto boots up after a timeout - or there is some slower polling
            # going on to wake up the controller.)
            with s_push.connect(PUSH_URL):
                print("client: A_CONTROLLER_END")
                s_push.send_string("A_CONTROLLER_END")
            with s_pull.bind(PULL_URL):
                reply = s_pull.recv_string()
                print("worker: " + reply)
                assert reply == "C_STOPPED"

            print(">>> TEST control message after worker died")
            # !!! SHOULD ENDLESSLY BLOCK !!!
            # send control message AFTER worker died
            with s_push.connect(PUSH_URL):
                print("client: A_CONTROLLER_STATE")
                s_push.send_string("A_CONTROLLER_STATE")
            # ---> sleep <---
            time.sleep(0.1)
            # wait before receiving for initialization
            with s_pull.bind(PULL_URL):
                reply = s_pull.recv_string()
                print("worker: " + reply)
                assert reply == "C_OK"
