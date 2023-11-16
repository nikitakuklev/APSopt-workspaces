import sys
import time
from pathlib import Path
from datetime import datetime
import json
import time
from collections import defaultdict

import numpy as np
from epics import PV
import debugpy

# PUT_ONLINE = False
PUT_ONLINE = True

if not PUT_ONLINE:
    PV.put = print

from common import (
    apply_scalar_pv_change,
    turn_on_pv_monitor,
    turn_off_pv_monitor,
    add_callback,
)
from bunch_cleaning import turn_on_BxB_feedback, turn_off_BxB_feedback


PINGER_KV_MAX_INCREMENT = 0.5
HPINGER_KV_MAX = 3.0
VPINGER_KV_MAX = 3.0

HPIN_BUCKET0_DELAY_NS = 800
VPIN_BUCKET0_DELAY_NS = 600

KICKOUT_VPIN_KV = 2.5
KICKOUT_VPIN_DELAY_NS = dict(
    inj_eff_bunches=2100,  # inj. eff. bunches near Bucket 620
    lifetime_bunches=600,  # inj. eff. bunches near Bucket 1280
)

PV_STRS = json.loads((Path(__file__).parent / "pvs_injection.json").read_text())
PVS = {k: PV(pv_str, auto_monitor=False) for k, pv_str in PV_STRS.items()}

# Storage used by callbacks
CB_DATA = defaultdict(dict)
CB_ARGS = defaultdict(dict)

INJ_KICKER_SETTINGS = dict(
    good_inj_filepath=Path(__file__).parent
    / "20231112T121651_inj_kicker_settings.json",
    poor_inj_filepath=Path(__file__).parent / "poor_inj_kicker_settings.json",
)

FIXED_GUN_SETTINGS = dict(MBM_volt=29.0, grid_volt=59.0)

INJ_BUCKET_FOR_INJ_EFF_MEAS = 620


def save_inj_kicker_settings_to_file():
    filepath = Path(f"{datetime.now():%Y%m%dT%H%M%S}_inj_kicker_settings.json")

    d = {}

    for k in [f"inj_kicker_{kicker_num}_kV_SP" for kicker_num in range(1, 4 + 1)]:
        pv = PVS[k]
        d[k] = pv.get()

    filepath.write_text(json.dumps(d, indent=2))

    return filepath


def load_inj_kicker_settings_from_file(filepath: Path):
    saved = json.loads(filepath.read_text())

    for k, v in saved.items():
        pv = PVS[k]
        pv.put(v)
        pv.get()


def restore_good_inj_kicker_settings():
    load_inj_kicker_settings_from_file(INJ_KICKER_SETTINGS["good_inj_filepath"])


def restore_poor_inj_kicker_settings():
    # load_inj_kicker_settings_from_file(INJ_KICKER_SETTINGS["poor_inj_filepath"])
    scale_inj_kickers(0.7)


def ramp_pinger(target_hpin_kV=None, target_vpin_kV=None, step_wait=2.0):
    """"""

    if (target_hpin_kV is not None) and (target_vpin_kV is not None):
        assert 0.0 <= target_hpin_kV <= HPINGER_KV_MAX
        assert 0.0 <= target_vpin_kV <= VPINGER_KV_MAX

        curr_hpin_kV_rb, curr_vpin_kV_rb = [
            PVS[k].get() for k in ["HPIN_KV_SP", "VPIN_KV_SP"]
        ]
        if curr_hpin_kV_rb < 0.0:
            curr_hpin_kV_rb = 0.0
        if curr_vpin_kV_rb < 0.0:
            curr_vpin_kV_rb = 0.0

        if target_hpin_kV > curr_hpin_kV_rb:  # Go up for H-pinger
            harray = np.arange(0.0, target_hpin_kV, PINGER_KV_MAX_INCREMENT)
            if target_hpin_kV not in harray:
                harray = np.append(harray, target_hpin_kV)

            first_ind = np.where(harray >= curr_hpin_kV_rb)[0][0]
            harray = harray[first_ind:]

        else:  # Go down for H-pinger
            harray = np.arange(target_hpin_kV, curr_hpin_kV_rb, PINGER_KV_MAX_INCREMENT)

            harray = harray[::-1]

        if target_vpin_kV > curr_vpin_kV_rb:  # Go up for V-pinger
            varray = np.arange(0.0, target_vpin_kV, PINGER_KV_MAX_INCREMENT)
            if target_vpin_kV not in varray:
                varray = np.append(varray, target_vpin_kV)

            first_ind = np.where(varray >= curr_vpin_kV_rb)[0][0]

            varray = varray[first_ind:]

        else:  # Go down for V-pinger
            varray = np.arange(target_vpin_kV, curr_vpin_kV_rb, PINGER_KV_MAX_INCREMENT)

            varray = varray[::-1]

        if len(harray) == len(varray):
            pass
        elif len(harray) > len(varray):
            if len(varray) != 0:
                varray = np.append(
                    varray, np.ones(len(harray) - len(varray)) * varray[-1]
                )
            else:
                varray = [PVS["VPIN_KV_SP"].get()] * len(harray)
        else:
            if len(harray) != 0:
                harray = np.append(
                    harray, np.ones(len(varray) - len(harray)) * harray[-1]
                )
            else:
                harray = [PVS["HPIN_KV_SP"].get()] * len(varray)

        assert len(harray) == len(varray)
        nSteps = len(harray)

        for iStep, (h, v) in enumerate(zip(harray, varray)):
            PVS["HPIN_KV_SP"].put(h)
            PVS["VPIN_KV_SP"].put(v)
            PVS["HPIN_KV_SP"].get()
            PVS["VPIN_KV_SP"].get()
            if iStep != nSteps - 1:
                time.sleep(step_wait)

    elif target_hpin_kV is not None:
        assert 0.0 <= target_hpin_kV <= HPINGER_KV_MAX

        curr_hpin_kV_rb = PVS["HPIN_KV_SP"].get()
        if curr_hpin_kV_rb < 0.0:
            curr_hpin_kV_rb = 0.0

        if target_hpin_kV > curr_hpin_kV_rb:  # Go up for H-pinger
            harray = np.arange(0.0, target_hpin_kV, PINGER_KV_MAX_INCREMENT)
            if target_hpin_kV not in harray:
                harray = np.append(harray, target_hpin_kV)

            first_ind = np.where(harray >= curr_hpin_kV_rb)[0][0]
            harray = harray[first_ind:]

        else:  # Go down for H-pinger
            harray = np.arange(target_hpin_kV, curr_hpin_kV_rb, PINGER_KV_MAX_INCREMENT)

            harray = harray[::-1]

        nSteps = len(harray)

        for iStep, h in enumerate(harray):
            PVS["HPIN_KV_SP"].put(h)
            PVS["HPIN_KV_SP"].get()
            if iStep != nSteps - 1:
                time.sleep(step_wait)

    elif target_vpin_kV is not None:
        assert 0.0 <= target_vpin_kV <= VPINGER_KV_MAX

        curr_vpin_kV_rb = PVS["VPIN_KV_SP"].get()
        if curr_vpin_kV_rb < 0.0:
            curr_vpin_kV_rb = 0.0

        if target_vpin_kV > curr_vpin_kV_rb:  # Go up for V-pinger
            varray = np.arange(0.0, target_vpin_kV, PINGER_KV_MAX_INCREMENT)
            if target_vpin_kV not in varray:
                varray = np.append(varray, target_vpin_kV)

            first_ind = np.where(varray >= curr_vpin_kV_rb)[0][0]
            varray = varray[first_ind:]

        else:  # Go down for V-pinger
            varray = np.arange(target_vpin_kV, curr_vpin_kV_rb, PINGER_KV_MAX_INCREMENT)

            varray = varray[::-1]

        nSteps = len(varray)

        for iStep, v in enumerate(varray):
            PVS["VPIN_KV_SP"].put(v)
            PVS["VPIN_KV_SP"].get()
            if iStep != nSteps - 1:
                time.sleep(step_wait)

    else:
        print(
            "Both `target_hpin_kV` and `target_vpin_kV` are None. No pinger ramp occurs."
        )


def ping_and_wait_till_fire(max_wait=5.0):
    """"""

    ping_pv = PVS["PING_CMD"]
    counter_pv = PVS["PING_COUNTER"]

    old_counter = counter_pv.get()

    ping_pv.put(1)
    ping_pv.get()

    # Wait for actual pinger firing
    while counter_pv.get() == old_counter:
        time.sleep(0.1)


def save_pinger_settings_to_file():
    filepath = Path(f"{datetime.now():%Y%m%dT%H%M%S}_pinger_settings.json")

    d = {}

    for k in [
        "PING_PLANE_SEL",
        "HPIN_KV_SP",
        "HPIN_NS_DELAY_SP",
        "VPIN_KV_SP",
        "VPIN_NS_DELAY_SP",
    ]:
        pv = PVS[k]
        d[k] = pv.get()

    filepath.write_text(json.dumps(d, indent=2))

    return filepath


def load_pinger_settings_from_file(
    filepath: Path, discharge=True, pinger_increment_step_wait=2.0
):
    saved = json.loads(filepath.read_text())

    kwargs_ramp = dict(
        target_hpin_kV=saved["HPIN_KV_SP"],
        target_vpin_kV=saved["VPIN_KV_SP"],
        step_wait=pinger_increment_step_wait,
    )
    ramp_pinger(**kwargs_ramp)

    if discharge:
        print("Discharging pingers...")
        ping_and_wait_till_fire(max_wait=5.0)
        ping_and_wait_till_fire(max_wait=5.0)
        print("Finished discharge.")

    for k, v in saved.items():
        if k not in ("HPIN_KV_SP", "VPIN_KV_SP"):
            pv = PVS[k]
            pv.put(v)
            pv.get()


def set_hpinger_delay_ns(delay_val):
    delay_pv = PVS["HPIN_NS_DELAY_SP"]

    if delay_pv.get() != delay_val:
        delay_pv.put(delay_val)
        delay_pv.get()


def set_vpinger_delay_ns(delay_val):
    delay_pv = PVS["VPIN_NS_DELAY_SP"]

    if delay_pv.get() != delay_val:
        delay_pv.put(delay_val)
        delay_pv.get()


def set_hpinger_delay_bucket(pinger_peak_bucket_index):
    delay_val = HPIN_BUCKET0_DELAY_NS + 2 * pinger_peak_bucket_index
    set_hpinger_delay_ns(delay_val)


def set_vpinger_delay_bucket(pinger_peak_bucket_index):
    delay_val = VPIN_BUCKET0_DELAY_NS + 2 * pinger_peak_bucket_index
    set_vpinger_delay_bucket(delay_val)


def set_pinger_plane(new_plane):
    avail_opts = ["", "v", "h", "hv"]
    assert new_plane in avail_opts

    plane_sel_pv = PVS["PING_PLANE_SEL"]

    cur_plane = plane_sel_pv.get()

    if new_plane != cur_plane:
        plane_sel_pv.put(avail_opts.index(new_plane))
        plane_sel_pv.get()


def setup_kickout_pinger_settings():
    set_pinger_plane("v")

    kwargs_ramp = dict(
        target_vpin_kV=KICKOUT_VPIN_KV,
        step_wait=2.0,
    )
    ramp_pinger(**kwargs_ramp)


def kickout_beam(max_wait=5.0):
    ping_and_wait_till_fire(max_wait=max_wait)


def kickout_lifetime_bunches(max_wait=5.0):
    set_vpinger_delay_ns(KICKOUT_VPIN_DELAY_NS["lifetime_bunches"])
    ping_and_wait_till_fire(max_wait=max_wait)


def kickout_inj_eff_bunches(init_dcct_val, max_wait=5.0):
    set_vpinger_delay_ns(KICKOUT_VPIN_DELAY_NS["inj_eff_bunches"])
    margin = 0.1  # [mA]
    while PVS["DCCT_2"].get() > init_dcct_val + margin:
        ping_and_wait_till_fire(max_wait=max_wait)


def meas_inj_eff_v0(n_shots=3, clear_beam_mode=3):
    """Based on meas_OnDemand_inj_eff() defined in 20171008_Tune_Scan_vs_Lifetime_and_InjEff.ipynb

    # nsls2.sr.pinger.VPINGER_KV_MAX = 2.3
    # # Empirically found to be the right with Bucket @ 620 (= 1280 - 1320/2) & vertical 1840 ns delay
    # # This may change if the ID gaps are changed. Right now they are closed

    nsls2.sr.pinger.HPINGER_KV_MAX = 2.8
    # Empirically found to be the right with Bucket @ 620 (= 1280 - 1320/2) & horizontal 2040 ns delay
    # This should not depend on ID gap states.

    clear_beam_mode: 0 = No clearning
                     1 = clear before the first shot
                     2 = clear after the last shot
                     3 = clear after each shoot
    """

    assert clear_beam_mode in [0, 1, 2, 3]

    restore_poor_inj_kicker_settings()

    stop_filepath = Path("STOP_INJ_EFF_FUNC")

    try:
        stop_filepath.unlink()
    except:
        pass
    print(
        '### Create a file named "STOP_INJ_EFF_FUNC" in the current directory to stop this function'
    )
    sys.stdout.flush()

    inj_eff_pv = PVS["inj_eff"]
    inject_pv = PVS["inject"]

    if clear_beam_mode:
        print("* Clearing beam")
        kickout_beam()
        time.sleep(1.0)

    ts_array = []
    inj_eff_array = []
    for iShot in range(n_shots):
        print(f"* Shot #{iShot+1}")

        ini_val = inj_eff_pv.get()
        ini_ts = inj_eff_pv.timestamp
        print(f"initial value = {ini_val:.3f}")

        inject_pv.put(1, wait=True)
        inject_pv.get()
        valid, new_val, new_timestamp = _wait_for_inj_eff_pv_update(ini_ts)

        if not valid:  # Second try
            print("* Injection command may not have worked. Re-trying.")
            inject_pv.put(1, wait=True)
            inject_pv.get()
            valid, new_val, new_timestamp = _wait_for_inj_eff_pv_update(ini_ts)

        if valid:
            ts_array.append(new_timestamp)
            new_ts_str = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(ts_array[-1])
            )
            inj_eff_array.append(new_val)
            print(f"{new_ts_str} :::: {new_val:.3f}%")
            sys.stdout.flush()

            if (clear_beam_mode == 3) or (
                (clear_beam_mode == 2) and (iShot == n_shots - 1)
            ):
                kickout_beam()
                time.sleep(1.0)
        else:
            ts_array.append(datetime.now().timestamp())
            # ts_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts_array[-1]))
            inj_eff_array.append(0.0)
            print(f"{ts_str} :::: {0.0:.3f}%")
            sys.stdout.flush()

        if (iShot != n_shots - 1) and stop_filepath.exists():
            print("Stop request detected. Aborting measurements.")
            break

    try:
        stop_filepath.unlink()
    except:
        pass

    return dict(timestamp=np.array(ts_array), inj_eff=np.array(inj_eff_array))


def _wait_for_inj_eff_pv_update(ini_timestamp, n_wait=3, sleep=2.0):
    inj_eff_pv = PVS["inj_eff"]

    updated = False
    for _ in range(n_wait):
        new_val = inj_eff_pv.get()
        new_timestamp = inj_eff_pv.timestamp
        if new_timestamp == ini_timestamp:
            print(f"new value (no change) = {new_val:.3f}")
            time.sleep(sleep)
        else:
            print(f"new value (with change) = {new_val:.3f}")
            valid = True
            break

    return updated, new_val, new_timestamp


def change_inj_target_bucket(target_bucket_number):
    assert 1 <= target_bucket_number <= 1320

    target_bucket_index = target_bucket_number - 1

    pvsp = PVS["inj_target_bucket_SP"]
    pvrb = PVS["inj_target_bucket_RB"]
    pv_apply = PVS["apply_target_bucket_change"]

    while pvrb.get() != target_bucket_index:
        pvsp.put(target_bucket_index)
        pvsp.get()
        pv_apply.put(1)
        pv_apply.get()
        time.sleep(2.0)


def change_gun_settings(gun_enabled, target_bucket_number, cw=False, pulse_width_ns=40):
    apply_scalar_pv_change(PVS["gun_enable_SP"], gun_enabled, sleep=1.0)
    assert PVS["gun_enable_RB"].get() == gun_enabled  # 0 = disabled; 1 = enabled

    if cw:
        new_SP = 0
    else:
        new_SP = 1  # single
    apply_scalar_pv_change(PVS["inj_trigger_mode"], new_SP, sleep=1.0)

    MBM_volt = FIXED_GUN_SETTINGS["MBM_volt"]
    grid_volt = FIXED_GUN_SETTINGS["grid_volt"]

    apply_scalar_pv_change(PVS["inj_divider_enable"], 0, sleep=1.0)  # 0 = disabled
    apply_scalar_pv_change(PVS["pulse_width"], pulse_width_ns, sleep=1.0)
    apply_scalar_pv_change(PVS["MBM_volt"], MBM_volt, sleep=1.0)
    apply_scalar_pv_change(PVS["grid_volt"], grid_volt, sleep=1.0)
    apply_scalar_pv_change(PVS["gun_pulse_mode"], 0, sleep=1.0)  # 0 = MBM; 1 = SBM

    change_inj_target_bucket(target_bucket_number)


def _wait_for_injection(prev_count, max_wait=120.0):
    t0 = time.perf_counter()
    timed_out = False
    while PVS["count_with_beam"].get() == prev_count:
        if time.perf_counter() - t0 > max_wait:
            timed_out = True
            break
        else:
            time.sleep(1.0)

    return timed_out


def _inject_up_to(
    inject_pv, mA_pv, target_mA, slee_after_inj=1.0, second_try_wait=3.0, max_wait=120.0
):
    restore_good_inj_kicker_settings()

    t0 = time.perf_counter()
    timed_out = False
    current_mA = mA_pv.get()
    if ("PICO" in mA_pv.pvname) or ("Camshaft" in mA_pv.pvname):
        try:
            current_mA = current_mA[1280]  # camshaft bunch
        except IndexError:
            current_mA = 0.0
    while current_mA < target_mA:
        ini_count = PVS["count_with_beam"].get()

        inject_pv.put(1, wait=True)
        inject_pv.get()

        inj_timed_out = _wait_for_injection(ini_count, max_wait=second_try_wait)

        if inj_timed_out:
            inject_pv.put(1, wait=True)
            inject_pv.get()

            inj_timed_out = _wait_for_injection(ini_count, max_wait=second_try_wait)

            if inj_timed_out:
                raise RuntimeError("Injection is not happening")

        if time.perf_counter() - t0 > max_wait:
            timed_out = True
            break

        time.sleep(slee_after_inj)

        current_mA = mA_pv.get()
        if ("PICO" in mA_pv.pvname) or ("Camshaft" in mA_pv.pvname):
            try:
                current_mA = current_mA[1280]  # camshaft bunch
            except IndexError:
                current_mA = 0.0

    return timed_out


def inject_single_shots_up_to(
    target_dcct_mA, slee_after_inj=1.0, second_try_wait=3.0, max_wait=120.0
):
    mA_pv = PVS["DCCT_2"]
    inject_pv = PVS["inject"]

    timed_out = _inject_up_to(
        inject_pv,
        mA_pv,
        target_dcct_mA,
        slee_after_inj=slee_after_inj,
        second_try_wait=second_try_wait,
        max_wait=max_wait,
    )

    return timed_out


def inject_camshaft_up_to(
    target_bunch_mA, slee_after_inj=1.0, second_try_wait=3.0, max_wait=120.0
):
    # TOFIX: Make sure to enable gun

    apply_scalar_pv_change(PVS["camshaft_bucket_index"], 1280, sleep=1.0)

    # mA_pv = PVS["camshaft_mA"]
    mA_pv = PVS["picoharp_camshaft_mA"]

    inject_pv = PVS["inject_camshaft"]

    timed_out = _inject_up_to(
        inject_pv,
        mA_pv,
        target_bunch_mA,
        slee_after_inj=slee_after_inj,
        second_try_wait=second_try_wait,
        max_wait=max_wait,
    )

    return timed_out


def refill_lifetime_meas_bunches(
    target_dcct_mA_list, target_bucket_number_list, pulse_width_ns_list
):
    assert (
        len(target_dcct_mA_list)
        == len(target_bucket_number_list)
        == len(pulse_width_ns_list)
    )
    for target_dcct_mA, target_bucket_number, pulse_width_ns in zip(
        target_dcct_mA_list, target_bucket_number_list, pulse_width_ns_list
    ):
        setup_inj_kickers_for_lifetime_meas(
            target_bucket_number, pulse_width_ns=pulse_width_ns
        )

        inject_single_shots_up_to(
            target_dcct_mA, slee_after_inj=1.0, second_try_wait=3.0, max_wait=120.0
        )


def setup_inj_kickers_for_inj_eff_meas(pulse_width_ns=40):
    gun_enabled = 0  # disable gun
    cw = True
    change_gun_settings(
        gun_enabled,
        INJ_BUCKET_FOR_INJ_EFF_MEAS,
        cw=cw,
        pulse_width_ns=pulse_width_ns,
    )

    restore_poor_inj_kicker_settings()


def cleanup_inj_kickers_for_inj_eff_meas():
    gun_enabled = 0  # disable gun
    cw = False
    change_gun_settings(
        gun_enabled,
        INJ_BUCKET_FOR_INJ_EFF_MEAS,
        cw=cw,
        pulse_width_ns=PVS["pulse_width"].get(),
    )


def setup_inj_kickers_for_lifetime_meas(target_bucket_number, pulse_width_ns=40):
    gun_enabled = 1  # enable gun
    cw = False

    change_gun_settings(
        gun_enabled,
        target_bucket_number,
        cw=cw,
        pulse_width_ns=pulse_width_ns,
    )

    restore_good_inj_kicker_settings()


def turn_on_cw_inj_for_inj_eff_meas():
    # Verify injection is in CW mode
    assert PVS["inj_trigger_mode"].get() == 0

    # Enable gun
    apply_scalar_pv_change(PVS["gun_enable_SP"], 1, sleep=1.0)

    # Verifiy gun is enabled
    assert PVS["gun_enable_RB"].get() == 1


def turn_off_cw_inj_for_inj_eff_meas():
    # Disable gun
    apply_scalar_pv_change(PVS["gun_enable_SP"], 0, sleep=1.0)

    # Verifiy gun is disabled
    assert PVS["gun_enable_RB"].get() == 0


def get_revolution_period():
    rf_freq_Hz = PVS["RF_freq_RB_Hz"].get()
    harmonic_number = 1320
    t_rev_second = 1 / (rf_freq_Hz / harmonic_number)

    return t_rev_second


def update_revolution_period():
    global T_REV
    T_REV = get_revolution_period()


def charge_to_ring_current(charge_nC, t_rev_second=None):
    if t_rev_second is None:
        t_rev_second = get_revolution_period()

    ring_current_mA = (charge_nC * 1e-9) / t_rev_second * 1e3

    return ring_current_mA


def ring_current_to_charge(ring_current_mA, t_rev_second=None):
    if t_rev_second is None:
        t_rev_second = get_revolution_period()

    charge_nC = (ring_current_mA * 1e-3) * t_rev_second * 1e9

    return charge_nC


def reset_callback_data(callback_func):
    if callback_func == callback_BTS_ICT2:
        CB_DATA["callback_BTS_ICT2"] = dict(
            mA=[], cum_mA=0.0, timestamp=[], stop_monitor=False
        )
    elif callback_func in (callback_DCCT_1, callback_DCCT_2, callback_DCCT_precise):
        CB_DATA[callback_func.__name__] = dict(mA=[], timestamp=[])
    else:
        raise NotImplementedError


def callback_BTS_ICT2(pvname, **kwargs):
    # This PV updates every 1 second.

    if False:
        debugpy.debug_this_thread()  # Needed for VSCode thread debugging
        print(kwargs)

    this_func_name = callback_BTS_ICT2.__name__
    cb_args = CB_ARGS[this_func_name]
    cb_data = CB_DATA[this_func_name]

    max_cum_mA = cb_args.get("max_cum_mA", 2.0)
    ICT_noise_level_nC = cb_args.get("ICT_noise_level_nC", 0.02)

    nC = kwargs.get("value")

    if nC > ICT_noise_level_nC:
        mA = charge_to_ring_current(nC, t_rev_second=T_REV)
    else:
        mA = 0.0

    timestamp = kwargs.get("timestamp")

    cb_data["mA"].append(mA)
    cb_data["cum_mA"] += mA
    cb_data["timestamp"].append(timestamp)

    if cb_data["cum_mA"] > max_cum_mA:
        cb_data["stop_monitor"] = True


def callback_DCCT_1(pvname, **kwargs):
    # This PV updates every 1 second.

    this_func_name = callback_DCCT_1.__name__
    _update_DCCT_callback_data(CB_DATA[this_func_name], kwargs)


def callback_DCCT_2(pvname, **kwargs):
    # This PV updates every 1 second.

    this_func_name = callback_DCCT_2.__name__
    _update_DCCT_callback_data(CB_DATA[this_func_name], kwargs)


def callback_DCCT_precise(pvname, **kwargs):
    # This PV updates every 5 seconds.

    this_func_name = callback_DCCT_precise.__name__
    _update_DCCT_callback_data(CB_DATA[this_func_name], kwargs)


def _update_DCCT_callback_data(cb_data, kwargs):
    mA = kwargs.get("value")
    timestamp = kwargs.get("timestamp")

    cb_data["mA"].append(mA)
    cb_data["timestamp"].append(timestamp)


def meas_inj_eff_v1(
    max_cum_mA=2.0, max_duration=60.0, pre_inj_wait=0.5, post_inj_wait=1.5
):
    """
    Must call setup_kickout_pinger_settings() first before this function is called.
    """

    setup_inj_kickers_for_inj_eff_meas()

    CB_ARGS["callback_BTS_ICT2"] = {
        "max_cum_mA": max_cum_mA,
        "ICT_noise_level_nC": 0.02,
    }
    add_callback(PVS["inj_charge_nC"], callback_BTS_ICT2)

    add_callback(PVS["DCCT_1"], callback_DCCT_1)
    add_callback(PVS["DCCT_2"], callback_DCCT_2)
    add_callback(PVS["DCCT_precise"], callback_DCCT_precise)

    reset_callback_data(callback_BTS_ICT2)
    reset_callback_data(callback_DCCT_1)
    reset_callback_data(callback_DCCT_2)
    reset_callback_data(callback_DCCT_precise)

    monitored_pv_keys = ["inj_charge_nC", "DCCT_1", "DCCT_2", "DCCT_precise"]
    for k in monitored_pv_keys:
        turn_on_pv_monitor(PVS[k])

    time.sleep(pre_inj_wait)

    init_dcct_val = PVS["DCCT_2"].get()

    turn_on_cw_inj_for_inj_eff_meas()

    reached_target_inj_charge = True

    t0 = time.perf_counter()
    while not CB_DATA["callback_BTS_ICT2"]["stop_monitor"]:
        if time.perf_counter() - t0 > max_duration:
            reached_target_inj_charge = False
            break

        time.sleep(1.0)

    turn_off_cw_inj_for_inj_eff_meas()

    time.sleep(post_inj_wait)

    if False:
        cb_data = CB_DATA["callback_DCCT_precise"]
        prev_len = len(cb_data["mA"])

        while prev_len == len(cb_data["mA"]):
            time.sleep(0.2)

    for k in monitored_pv_keys:
        turn_off_pv_monitor(PVS[k])

    total_inj_charge_mA = CB_DATA["callback_BTS_ICT2"]["cum_mA"]

    cb_data = CB_DATA["callback_DCCT_2"]
    dcct_change_mA = cb_data["mA"][-1] - cb_data["mA"][0]

    efficiency = dcct_change_mA / total_inj_charge_mA

    kickout_inj_eff_bunches(init_dcct_val, max_wait=5.0)

    cleanup_inj_kickers_for_inj_eff_meas()

    return dict(
        eff_percent=efficiency * 1e2,
        reached_target_inj_charge=reached_target_inj_charge,
    )


def scale_inj_kickers(scaling):
    good = json.loads(INJ_KICKER_SETTINGS["good_inj_filepath"].read_text())

    for k, v in good.items():
        pv = PVS[k]
        pv.put(v * scaling)
        pv.get()


T_REV = get_revolution_period()

if __name__ == "__main__":
    if False:
        save_inj_kicker_settings_to_file()

    elif False:
        target_bunch_mA = 0.05  # 0.4
        inject_camshaft_up_to(target_bunch_mA, slee_after_inj=1.0)

    elif False:
        setup_kickout_pinger_settings()
        kickout_lifetime_bunches(max_wait=5.0)

    elif True:
        target_bunch_mA = 0.4
        inject_camshaft_up_to(target_bunch_mA, slee_after_inj=1.0)

        FIXED_GUN_SETTINGS["MBM_volt"] = 29.0
        FIXED_GUN_SETTINGS["grid_volt"] = 59.0

        # target_dcct_mA_list = [10.0, 20.0]
        target_dcct_mA_list = [7.5, 15.0]
        target_bucket_number_list = [1300, 20]  # [1281, 0]
        pulse_width_ns_list = [40.0, 40.0]
        refill_lifetime_meas_bunches(
            target_dcct_mA_list, target_bucket_number_list, pulse_width_ns_list
        )

    elif True:
        res = meas_inj_eff_v1(
            max_cum_mA=5.0, max_duration=60.0, pre_inj_wait=0.5, post_inj_wait=1.5
        )
        print(res)
