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
    for family, K2L in inputs_dict.items():
        sexts_ctrl.change_sext_strengths(K2L, family)

    time.sleep(3.0)  # TOFIX: check RB-SP diff


def setup_for_orbit_correction():
    """Set current SA orbit as reference orbit for SOFB
    => TOFIX: I need to set to BBA orbit
    """
    PVS["SOFB_ref_orb_reset"].put(1)
    PVS["SOFB_ref_orb_reset"].get()

    apply_scalar_pv_change(PVS["SOFB_cor_frac"], 0.10)
    apply_scalar_pv_change(PVS["SOFB_adaptive"], 0.0)


def turn_on_SOFB():
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


def turn_on_tune_feedback():
    while PVS["tune_fb_enabled"].get() == 0:
        PVS["tune_fb_ena_set"].put(1)
        PVS["tune_fb_ena_set"].get()
        time.sleep(3.0)


def turn_off_tune_feedback():
    return  # TOFIX

    while PVS["tune_fb_enabled"].get() == 1:
        PVS["tune_fb_ena_set"].put(0)
        PVS["tune_fb_ena_set"].get()
        time.sleep(3.0)


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


def correct_tunes(dnu_thresh=2e-3):
    while (np.abs(PVS["nux_RB"].get() - PVS["nux_SP"].get()) > dnu_thresh) or (
        np.abs(PVS["nuy_RB"].get() - PVS["nuy_SP"].get()) > dnu_thresh
    ):
        PVS["cor_tunes"].put(1)
        PVS["cor_tunes"].get()
        time.sleep(4.0)


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
    target_bucket_number_list = [1300, 20]  # [1281, 0]
    pulse_width_ns_list = [40.0, 40.0]
    injection.refill_lifetime_meas_bunches(
        target_dcct_mA_list, target_bucket_number_list, pulse_width_ns_list
    )


def master_eval_function(inputs_dict):
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
    *) Turn on tune feedback [TODO: function not working]
    *) Inject 20 mA of "lifetime bunches"
    *) Set SOFB refence orbit to BBA values and set other SOFB settings
      - Call `setup_for_orbit_correction()`
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

    turn_on_BxB_feedback()

    time.sleep(2.0)
    res = meas_beam_props_w_bxb_on(duration=1.0)
    nux = res["nux_RB_avg"]
    nuy = res["nuy_RB_avg"]
    res["eps_x_nm_avg"] *= 1e-9  # converte [nm] to [m]
    res["eps_y_pm_avg"] *= 1e-12  # converte [pm] to [m]
    eps_x_bxbOn = res["eps_x_nm_avg"]
    eps_y_bxbOn = res["eps_y_pm_avg"]

    turn_off_BxB_feedback()

    if True:
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
            objective_lifetime = res["avg"]

            tau_suppl = res["moni_data"]
            eps_y_bxbOff = tau_suppl["eps_y_pm"]["value"] * 1e-12
            if np.max(eps_y_bxbOff) > 30e-12:
                print("* eps_y too large! Lifetime value set to NaN.")
                objective_lifetime = np.nan
        except:
            print(f"{traceback.format_exc()}")
            logger.error("Lifetime calc failed")
            logger.error(f"{traceback.format_exc()}")
            objective_lifetime = np.nan
        print(
            f"Lifetime = {objective_lifetime:.1f} (took {time.perf_counter()-t0:.1f} [s])"
        )

    turn_off_SOFB()

    if True:
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

        print("Eval. func. finished.")
        sys.stdout.flush()

        return {
            "LT": objective_lifetime,
            "EFF": objective_eff,
            "_nux": nux,
            "_nuy": nuy,
            "_eps_x_bxbOn": eps_x_bxbOn,
            "_eps_y_bxbOn": eps_y_bxbOn,
            # "_tau_suppl_data": tau_suppl,
        }

    else:
        print("Eval. func. finished.")
        sys.stdout.flush()

        return {
            "LT": objective_lifetime,
            "_nux": nux,
            "_nuy": nuy,
            "_eps_x_bxbOn": eps_x_bxbOn,
            "_eps_y_bxbOn": eps_y_bxbOn,
            # "_tau_suppl_data": tau_suppl,
        }


if __name__ == "__main__":
    if False:
        # Measure chromaticity [must use at least 5-sec dealy between
        # each step to allow 4-step BxB tune window]
        lin_ksix, lin_ksiy = meas_lin_chrom(max_wait=60.0)

    elif False:
        duration = 5.0  # [s]
        meas_beam_props_w_bxb_on(duration)

    elif False:
        setup_for_tune_correction()

        # turn_on_tune_feedback() # TODO: not working

    elif False:
        # TOFIX: set ref to BBA
        setup_for_orbit_correction()

    elif False:
        # TOFIX
        lin_ksix, lin_ksiy = meas_lin_chrom(max_wait=60.0)
        print(lin_ksix, lin_ksiy)

    elif False:
        prep_for_optim()

    elif True:
        inputs_dict = json.loads(
            Path(
                "/nsls2/users/yhidaka/git_repos/APSopt-workspaces/nsls-ii/scripts/test_inputs.json"
            ).read_text()
        )
        # inputs_dict['SM2'] = inputs_dict['SM2B']
        # del inputs_dict['SM2B']
        master_eval_function(inputs_dict)
