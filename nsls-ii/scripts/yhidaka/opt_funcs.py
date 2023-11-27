"""
Added packages ($ mamba install pkg_name -c conda-forge --strict-channel-priority):
pyepics, black
"""

import logging
import traceback
import time
import json
from pathlib import Path
from collections import defaultdict
import sys

import numpy as np
import matplotlib.pyplot as plt
from epics import PV

from common import (
    apply_scalar_pv_change,
    add_callback,
    turn_on_pv_monitor,
    turn_off_pv_monitor,
)
import lifetime
import injection
import sexts_ctrl
from bunch_cleaning import turn_on_BxB_feedback, turn_off_BxB_feedback

logger = logging.getLogger(__name__)


PV_STRS = json.loads((Path(__file__).parent / "pvs_opt_funcs.json").read_text())
PVS = {k: PV(pv_str, auto_monitor=False) for k, pv_str in PV_STRS.items()}

# Storage used by callbacks
CB_DATA = defaultdict(dict)
CB_ARGS = defaultdict(dict)


def setup_pinger_for_inj_eff_meas(pinger_peak_bucket_index, clear_setpoint_kV):
    # pinger_peak_bucket_index = 620  # = 1280 - 1320/2

    pinger_settings_filepath = injection.save_pinger_settings_to_file()
    print(f'Saved current pinger settings to "{pinger_settings_filepath}"')

    injection.set_vpinger_delay(pinger_peak_bucket_index)
    injection.set_pinger_plane("v")
    injection.ramp_pinger(target_vpin_kV=clear_setpoint_kV, step_wait=2.0)


def clearnup_pinger_for_inj_eff_meas(pinger_settings_filepath):
    injection.load_pinger_settings_from_file(pinger_settings_filepath, discharge=True)


def acquire_lifetime_normalization_data():
    # First inject into camshaft area
    # See in what range of DCCT, orbit is OK
    # Try cleaning bunches

    # Acquire DCCT, BPM attenuation, BPM sum signals

    raise NotImplementedError


def analyze_lifetime_normalization_data():
    raise NotImplementedError


def apply_sextupole_changes(inputs_dict):
    tStart = time.perf_counter()

    if False:
        for family, K2L in inputs_dict.items():
            sexts_ctrl.change_sext_strengths(
                K2L,
                family,
                max_dI=2.0,
                step_wait=1.0,
                wait_small_SP_RB_diff=False,
                max_SP_RB_dI=0.05,
            )  # TO-ADJUST
    else:
        K2L_list = []
        family_list = []
        for family, K2L in inputs_dict.items():
            K2L_list.append(K2L)
            family_list.append(family)
        sexts_ctrl.change_sext_strengths(
            K2L_list, family_list, max_dI=1.0, step_wait=1.0
        )  # TO-ADJUST

    print(f"Sextupole adj. took {time.perf_counter()-tStart:.1f}.")

    # time.sleep(3.0)
    time.sleep(1.0)


def setup_for_orbit_correction(bba_ref=True, use_adaptive=True):
    """Set reference orbit and other settings for SOFB"""

    if not bba_ref:
        pv = PVS["SOFB_ref_orb_reset"]
    else:
        pv = PVS["SOFB_ref_orb_zero"]
    pv.get()

    if not use_adaptive:
        apply_scalar_pv_change(PVS["SOFB_cor_frac"], 0.10)
        apply_scalar_pv_change(PVS["SOFB_adaptive"], 0)
    else:
        apply_scalar_pv_change(PVS["SOFB_adaptive"], 1)


def turn_on_SOFB():
    if lifetime.PVS["DCCT_1"].get() < 1.0:
        print("WARNING: No beam. Not turning on SOFB")
        return

    if PVS["SOFB_enabled"].get() != 1:
        print("SOFB not enabled yet. Enabling now!")
        PVS["SOFB_turn_on"].put(1)
        PVS["SOFB_turn_on"].get()


def turn_off_SOFB():
    if PVS["SOFB_enabled"].get() != 0:
        print("SOFB enabled. Disabling now!")
        PVS["SOFB_turn_off"].put(1)
        PVS["SOFB_turn_off"].get()


def correct_orbit(SOFB_off_on_exit, dx_rms_thresh=2e-6, dy_rms_thresh=2e-6):
    while PVS["SOFB_mon_enabled"].get() == 0:
        PVS["SOFB_mon_turn_on"].put(1)
        PVS["SOFB_mon_turn_on"].get()
        time.sleep(2.0)

    while True:
        dx_rms = PVS["SOFB_dx_rms"].get() * 1e-3
        dy_rms = PVS["SOFB_dy_rms"].get() * 1e-3
        print(f"RMS(dx, dy) [m] = ({dx_rms:.3e}, {dy_rms:.3e})")

        if (dx_rms > dx_rms_thresh) or (dy_rms > dy_rms_thresh):
            turn_on_SOFB()
            time.sleep(3.0)
        else:
            print("Orbit correction converged")
            break

    if SOFB_off_on_exit:
        turn_off_SOFB()


def _change_tune_feedback_status(turn_on):
    if turn_on:
        if PVS["tune_fb_enabled"].get() == 0:
            PVS["tune_fb_ena_set_1"].put(1)

            # The value you caput here doesn't matter. It always caput's the value of 1.
            # Without this caput, the feedback will not turn on.
            PVS["tune_fb_ena_set_2"].put(1)

            while PVS["tune_fb_enabled"].get() == 0:  # Wait unitl it turns on
                time.sleep(0.2)
    else:
        if PVS["tune_fb_enabled"].get() == 1:
            PVS["tune_fb_ena_set_1"].put(0)
            # Unlike the "turn-on" case, you don't have to caput PVS["tune_fb_ena_set_2"].

            while PVS["tune_fb_enabled"].get() == 1:  # Wait unitl it turns off
                time.sleep(0.2)


def turn_on_tune_feedback():
    _change_tune_feedback_status(True)


def turn_off_tune_feedback():
    _change_tune_feedback_status(False)


def setup_for_tune_correction():
    turn_off_tune_feedback()

    PVS["tune_fb_min_dcct"].put(0.3)
    PVS["tune_fb_min_dcct"].get()

    PVS["max_tune_delta"].put(0.1)
    PVS["max_tune_delta"].get()

    PVS["nux_SP"].put(0.22)
    PVS["nux_SP"].get()
    PVS["nuy_SP"].put(0.26)
    PVS["nuy_SP"].get()


def cleanup_for_tune_correction():
    turn_on_tune_feedback()

    PVS["max_tune_delta"].put(0.02)
    PVS["max_tune_delta"].get()

    PVS["nux_SP"].put(0.22)
    PVS["nux_SP"].get()
    PVS["nuy_SP"].put(0.267)
    PVS["nuy_SP"].get()


def _wait_for_bxb_tune_pv_update():
    rb_d = {k: PVS[k].get() for k in ["nux_RB", "nuy_RB"]}
    prev_rb_ts_d = {k: PVS[k].timestamp for k in list(rb_d)}

    rb_pv_updated = {k: False for k in list(rb_d)}

    print("Waiting for BxB tune PVs to update...", end=" ")
    while True:
        for k, _updated in rb_pv_updated.items():
            if not _updated:
                rb_d[k] = PVS[k].get()
                new_ts = PVS[k].timestamp
                if new_ts != prev_rb_ts_d[k]:
                    prev_rb_ts_d[k] = new_ts
                    rb_pv_updated[k] = True

        if all(list(rb_pv_updated.values())):
            break
        else:
            time.sleep(0.2)
    print("Done")

    return rb_d


def correct_tunes(dnu_thresh=2e-3):
    pv_d = {k: PVS[k] for k in ["nux_RB", "nux_SP", "nuy_RB", "nuy_SP"]}

    sp_d = {k: PVS[k].get() for k in ["nux_SP", "nuy_SP"]}

    while True:
        rb_d = _wait_for_bxb_tune_pv_update()

        if (np.abs(rb_d["nux_RB"] - sp_d["nux_SP"]) < dnu_thresh) and (
            np.abs(rb_d["nuy_RB"] - sp_d["nuy_SP"]) < dnu_thresh
        ):
            break
        else:
            PVS["cor_tunes"].put(1)
            time.sleep(1.0)

            while PVS["cor_tunes"].get() == 1:
                time.sleep(0.2)


def setup_for_chrom_meas():
    PVS["chrom_freq_LB"].put(-100)
    PVS["chrom_freq_LB"].get()

    PVS["chrom_freq_UB"].put(+100)
    PVS["chrom_freq_UB"].get()

    PVS["chrom_nsteps"].put(3)
    PVS["chrom_nsteps"].get()

    PVS["chrom_step_wait"].put(3.0)
    PVS["chrom_step_wait"].get()

    PVS["chrom_tune_src"].put(0)  # 0=BxB; 1=TbT
    PVS["chrom_tune_src"].get()

    PVS["chrom_fit_order"].put(1)
    PVS["chrom_fit_order"].get()


def cleanup_for_chrom_meas():
    PVS["chrom_fit_order"].put(2)
    PVS["chrom_fit_order"].get()


def meas_lin_chrom(max_wait=60.0):
    # TOFIX: it doesn't wait until measurement ends (Use message to detect it)

    ksix_pv = PVS["lin_ksi_x"]
    ksiy_pv = PVS["lin_ksi_y"]

    _ = ksix_pv.get()
    prev_ts = ksix_pv.timestamp

    PVS["chrom_meas_start"].put(1, wait=True)

    t0 = time.perf_counter()
    while ksix_pv.timestamp == prev_ts:
        time.sleep(2.0)
        lin_ksix = ksix_pv.get()
        lin_ksiy = ksiy_pv.get()

        if time.perf_counter() - t0 > max_wait:
            lin_ksix = np.nan
            lin_ksiy = np.nan
            break

    return lin_ksix, lin_ksiy


def callback_append_to_list(pvname, **kwargs):
    if False:
        debugpy.debug_this_thread()  # Needed for VSCode thread debugging
        print(kwargs)

    cb_data = CB_DATA[pvname]

    for k in ["value", "timestamp"]:
        if k not in cb_data:
            cb_data[k] = []
        cb_data[k].append(kwargs.get(k))


def meas_beam_props_w_bxb_on(duration):
    pv_keys = [
        "nux_RB",
        "nuy_RB",
        "eps_x_nm",
        "eps_y_pm",
        "bxb_SRAM_rms_x",
        "bxb_SRAM_rms_y",
    ]

    for k in pv_keys:
        add_callback(PVS[k], callback_append_to_list)
        turn_on_pv_monitor(PVS[k])

    time.sleep(duration)

    for k in pv_keys:
        turn_off_pv_monitor(PVS[k])

    res = {}
    for k in pv_keys:
        pvname = PV_STRS[k]
        if not k.startswith("bxb_SRAM_rms_"):
            res[k] = {k2: np.array(v2) for k2, v2 in CB_DATA[pvname].items()}
            res[f"{k}_avg"] = np.mean(res[k]["value"])
            del res[k]
        else:
            for k2, v2 in CB_DATA[pvname].items():
                if k2 == "value":
                    res[k2] = np.max(v2)
                elif k2 == "timestamp":
                    res[k2] = v2[0]  # just keep the timestamp for the first measurement
                else:
                    raise ValueError(k2)

        for k2, v2 in CB_DATA[pvname].items():
            v2.clear()

    return res


def prep_for_optim():
    injection.setup_kickout_pinger_settings()

    turn_off_BxB_feedback()

    target_bunch_mA = 0.4
    injection.inject_camshaft_up_to(target_bunch_mA, slee_after_inj=1.0)

    injection.FIXED_GUN_SETTINGS["MBM_volt"] = 29.0
    injection.FIXED_GUN_SETTINGS["grid_volt"] = 59.0

    target_dcct_mA_list = [10.0, 20.0]
    # target_bucket_number_list = [1300, 20]  # [1281, 0]
    target_bucket_number_list = [1300 - 5, 20 + 5]
    # pulse_width_ns_list = [40.0, 40.0]
    pulse_width_ns_list = [50.0, 50.0]
    injection.refill_lifetime_meas_bunches(
        target_dcct_mA_list, target_bucket_number_list, pulse_width_ns_list
    )

    # TODO: Do lattice correction with the actual fill pattern, not <2 mA

    # TODO: Set chrom to +3/+3
    # TODO: Add vertical dispersion to 15 pm

    # TODO: Set right BPM attenuattion (20-dB)

    # TODO: add automatic refill when beam is too low or just pring WARNING

    # TODO: lifetime -> add "max_samples" (or use "max_wait")

    # TODO: Restore all settings after the shift (RF freq, DCCT range,
    # tune FB target tunes, BBF soft trig., etc.)


def master_eval_function(
    inputs_dict, meas_bxb_tunes=False, meas_lifetime=False, meas_inj_eff=False
):
    """Evaluation function that maps variables to objectives

    Bucket index starts from 0, not 1.

    Vertical Pinger with 2.5 kV and 2100 ns delay:
    1) Can kick out beam placed around Bucket 620
      - Use Buckets 620-639 [40 ns] for injection efficiency measurements
    2) Can keep beam placed around Bucket 1280
       (These bunches will be refered to as "lifetime bunches" from this point on.)
      - Inject a total of 20 mA into 2 bunch trains for lifetime measurements:
        - Inject 10 mA into Buckets 1281-1300 [40 ns]
        - Leave gap at Buckets 1301-1319 [40 ns] (Bucket 1319 is the last bucket)
        - Inject 10 mA into Buckets 0-19 [40 ns]

    "injection efficiency mode":
    - Scale down injection kicker strengths
    - Adjust target bucket to 620

    "lifetime mode":
    - Restore injection kicker strengths for operation
    - Adjust target bucket to 1281 and then 0.
    - (optional) Run bunch cleaning after refill

    Preparation:
    *) Turn on pinger AC contactor
    *) Restore the previously saved bare MASAR orbit/lattice
    *) Cycle quads/sexts twice
    *) Turn on BPM auto gain control (AGC)
    *) Lower the DCCT range to 200 mA:
       $ caput SR:C03-BI{DCCT:1}Range-Sel 2
    *) [SKIP: already done]
       Save current (good) injection kicker settings to a file using
       `injection.save_inj_kicker_settings_to_file()`
    *) Put camshaft at Bucket 1280 to 0.4 mA
    *) Turn on bunch-by-bunch (BxB) feedback
    *) Turn on tune feedback
    *) Inject 20 mA of "lifetime bunches"
    *) Set SOFB refence orbit to BBA values and set other SOFB settings
      - Call `setup_for_orbit_correction(bba_ref=True, use_adaptive=True)`
    *) Turn on SOFB (high cor.frac, but not too high to slip away from target tunes)
    *) Adjust RF frequency to bring horizontal corrector sum to ~162 A (was ~174 A before).
    *) Adjust ROI and exposure time of the BMA pinhole camera
    *) Turn off slow orbit feedback (SOFB)
    *) Turn off tune feedback (TuneFB)
    *) Turn off BxB feedback (BBFB)
    *) Scrape beam down to ~2 mA
    *) Turn on pinger HVPS
    *) Correct linear optics, coupling, & vertical dispersion
    *) Adjust exposure time of pinhole to be able to see beam at ~2 mA
    *) Turn on BBFB
    *) If camshaft too low, refill camshaft
    *) Measure(+/-100 Hz, 5 steps / 5sec each)/adjust linear chromaticity to +3/+3
    *) Test chromaticity null knobs
    *) Turn off BBFB
    *) If necessary, re-correct linear optics, coupling, & vertical dispersion.
    *) Turn on BBFB
    *) Confirm linear chromaticity is still approx. +3/+3
    *) Turn off BBFB
    *) Note BPM attenuation values at ~2 mA => All 10 dB
    *) Re-adjust exposure time of pinhole for 20 mA.
    *) Refill to 20 mA, and measure BPM attenuation values.
    *) Decide the refill threshold beam current for "lifetime bunches" such
       that BPMs do not saturate with the fixed BPM attenuation values.
       => Set all to 20 dB with AGC off
    *) Set the fixed BPM attenuation values and turn off AGC.
    *) [SKIPPED] Measure a lifetime normalization curve, starting from 20 mA down to
       the refill threshold current:
      - Off: BBFB, TuneFB, AGC
      - On: SOFB
      - Acquire:
        - DCCT (3 types)
          - SR:C03-BI{DCCT:1}I:Real-I
          - SR:C03-BI{DCCT:1}I:Total-I (apparently same as "Real-I")
          - SR:C03-BI{DCCT:1}I:Total-I_ (most accurate one)
        - BPM sum signals (suffix "Ampl:SSA-Calc")
        - emittances
    *) Come up with a lifetime calibration formula and integrate it into the
       optimizer. => use I^(2/3)
    *) Kick out beam completely
    *) Set injector mode to "injection efficiency mode"
    *) Lower injection kicker settings until injection becomes poor.
       Decide the injection kicker settings to be used during optimization.
       Save these settings to a file using
       `injection.save_inj_kicker_settings_to_file()` and manually copy the
       file path to `INJ_KICKER_SETTINGS["poor_inj_filepath"]` in `injection.py`.
    *) Kick out beam completely

    Optimization Steps:
    1) Set pinger for kickout configuration
    2) Turn off BBFB
    3) Put camshaft at Bucket 1280
    4) Inject 20 mA of "lifetime bunches"
    ----------------------------------------
    5) Change sextupole settings
    6) Turn on SOFB and wait for orbit correction to complete
    7) Briefly turn on BBFB to capture tunes, emittances, BxB stability
    *) Turn off BBFB
    8) Start a lifetime measurement (BxB off)
      - capturing emittances
    9) Turn off SOFB
    10) Start an injection efficiency measurement (BxB off)
    11) If camshaft is too low (0.2 mA):
        - Click camshaft injection button to refill
    12) If "lifetime bunches" are too low (0.2 mA per bunch):
        - Switch injector mode to "lifetime mode" to refill to 20 mA
    13) Go back to Step 5 and repeat

    Backout Procedures:
    *) Turn on AGC
    *) Turn on BBFB
    *) Restore TuneFB settings (min DCCT = 1.0 mA, max tune delta = 0.02)
    *) Turn on TuneFB
    *) Restore DCCT range selection to "1 A"
       $ caput SR:C03-BI{DCCT:1}Range-Sel 1
    *) Restore original RF frequency
    *) Restore operation MASAR files
    *) Restore injeciton kicker settings
    *) Restore pinger nominal settings (0.2 kV and 800/60 ns delay.)
       and discharge.
    *) Turn off pinger PVHS


    """

    outputs = {}

    # Set sextupoles
    logger.info(f"Setting sextupoles to {inputs_dict}")

    inputs_dict_copy = {}
    for k, v in inputs_dict.items():
        if k == "SM2B":
            inputs_dict_copy["SM2"] = v
        else:
            inputs_dict_copy[k] = v

    apply_sextupole_changes(inputs_dict_copy)

    # Correct orbit (Thresholds in [m])
    SOFB_off_on_exit = False
    correct_orbit(SOFB_off_on_exit, dx_rms_thresh=10e-6, dy_rms_thresh=10e-6)

    if meas_bxb_tunes:
        turn_on_BxB_feedback()

        time.sleep(2.0)

        res = meas_beam_props_w_bxb_on(duration=1.0)

        nux = res["nux_RB_avg"]
        nuy = res["nuy_RB_avg"]
        outputs["_nux"] = nux
        outputs["_nuy"] = nuy

        res["eps_x_nm_avg"] *= 1e-9  # converte [nm] to [m]
        res["eps_y_pm_avg"] *= 1e-12  # converte [pm] to [m]
        eps_x_bxbOn = res["eps_x_nm_avg"]
        eps_y_bxbOn = res["eps_y_pm_avg"]
        outputs["_eps_x_bxbOn"] = eps_x_bxbOn
        outputs["_eps_y_bxbOn"] = eps_y_bxbOn

        turn_off_BxB_feedback()

    if meas_lifetime:
        # Lifetime
        logger.info("Lifetime measurement START")
        t0 = time.perf_counter()
        try:
            res = lifetime.measLifetimeAdaptivePeriod(
                max_wait=120.0,
                update_period=0.5,
                sigma_cut=3.0,
                sum_diff_thresh_fac=5.0,  # 10.0,
                min_samples=5,  # min. of 5 seconds
                abort_pv=None,
                mode="online",
                min_dcct_mA=0.2,
            )

            raw_tau_hr = res["avg"]

            # objective_lifetime = res["avg"]
            objective_lifetime = res["norm_tau"]

            tau_suppl = res["moni_data"]
            eps_x_bxbOff = np.median(tau_suppl["eps_x_nm"]["value"]) * 1e-9
            eps_y_bxbOff = np.median(tau_suppl["eps_y_pm"]["value"]) * 1e-12
            outputs["_eps_x_bxbOff"] = eps_x_bxbOff
            outputs["_eps_y_bxbOff"] = eps_y_bxbOff
            if eps_y_bxbOff > 30e-12:
                print("* eps_y too large! Lifetime value set to NaN.")
                objective_lifetime = np.nan
        except:
            print(f"{traceback.format_exc()}")
            logger.error("Lifetime calc failed")
            logger.error(f"{traceback.format_exc()}")
            objective_lifetime = np.nan
            raw_tau_hr = np.nan
            eps_y_bxbOff = np.nan
        print(
            f"Lifetime = {objective_lifetime:.3f} (raw tau [hr] = {raw_tau_hr:.3f}, epsy [pm] = {eps_y_bxbOff*1e12:.2f}) (took {time.perf_counter()-t0:.1f} [s])"
        )

        outputs["LT"] = objective_lifetime
        outputs["_raw_LT"] = res["avg"]

        # outputs["_tau_suppl_data"] = tau_suppl

    # turn_off_SOFB()

    if meas_inj_eff:
        # Injection efficiency
        logger.info("Efficiency measurement START")
        t0 = time.perf_counter()
        try:
            res = injection.meas_inj_eff_v1(
                max_cum_mA=2.0, max_duration=60.0, pre_inj_wait=2.0, post_inj_wait=3.0
            )
            objective_eff = res["eff_percent"]
        except:
            logger.error("Efficiency script failed")
            logger.error(f"{traceback.format_exc()}")
            objective_eff = np.nan
        print(
            f"Inj. Eff. = {objective_eff:.2f} (took {time.perf_counter()-t0:.1f} [s])"
        )

        outputs["EFF"] = objective_eff

    print("Eval. func. finished.")
    sys.stdout.flush()

    return outputs


if __name__ == "__main__":
    if False:
        # Measure chromaticity [must use at least 5-sec dealy between
        # each step to allow 4-step BxB tune window]
        lin_ksix, lin_ksiy = meas_lin_chrom(max_wait=60.0)

    elif False:
        duration = 5.0  # [s]
        meas_beam_props_w_bxb_on(duration)

    elif False:
        # turn_off_tune_feedback()
        # turn_on_tune_feedback()

        correct_tunes(1e-6)

    elif False:
        setup_for_tune_correction()

    elif False:
        setup_for_orbit_correction(bba_ref=True, use_adaptive=True)

    elif False:
        # TOFIX
        lin_ksix, lin_ksiy = meas_lin_chrom(max_wait=60.0)
        print(lin_ksix, lin_ksiy)

    elif False:
        prep_for_optim()

    elif True:
        # turn_off_BxB_feedback()
        turn_on_BxB_feedback()

    elif True:
        if False:
            inputs_dict = json.loads(
                Path(
                    "/nsls2/users/yhidaka/git_repos/APSopt-workspaces/nsls-ii/scripts/test_inputs.json"
                ).read_text()
            )
            # inputs_dict['SM2'] = inputs_dict['SM2B']
            # del inputs_dict['SM2B']

        inputs_dict = {
            "SL1": -2.8533953007500004,
            "SL2": 6.6004154682500005,
            "SL3": -6.334085031150001,
            "SH1": 3.6690887385,
            "SH3": -1.25884830815,
            "SH4": -3.40149365265,
            "SM1A": -4.27066061076,
            "SM1B": -5.48713351964,
            "SM2B": 7.295418874749999,
        }

        master_eval_function(inputs_dict, meas_lifetime=True)
