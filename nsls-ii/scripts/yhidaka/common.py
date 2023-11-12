import time


def apply_scalar_pv_change(pv, new_SP, sleep=1.0):
    while pv.get() != new_SP:
        pv.put(new_SP)
        time.sleep(sleep)


def turn_on_pv_monitor(pv):
    pv.auto_monitor = True


def turn_off_pv_monitor(pv):
    pv.auto_monitor = False


def add_callback(pv, new_callback_func):
    turn_off_pv_monitor(pv)

    for callback_index, (callback_func, kwargs) in pv.callbacks.items():
        if callback_func == new_callback_func:
            # `new_callback_func` has been already added. Not adding a duplicate
            return

    pv.add_callback(new_callback_func)
