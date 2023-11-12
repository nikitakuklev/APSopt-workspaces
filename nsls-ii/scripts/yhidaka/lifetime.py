import time
import json
from pathlib import Path
from collections import defaultdict
import re

import numpy as np
import matplotlib.pyplot as plt
from epics import PV

from common import add_callback, turn_on_pv_monitor, turn_off_pv_monitor

REGBPM_PV_PREFIXES = [
    f"SR:C{cell_num:02d}-BI{{BPM:{bpm_num:d}}}"
    for cell_num in range(1, 30 + 1)
    for bpm_num in range(1, 6 + 1)
]
REGBPM_SUM_PVS = [
    PV(f"{prefix}Ampl:SSA-Calc", auto_monitor=False) for prefix in REGBPM_PV_PREFIXES
]
REGBPM_SUMSTD_PVS = [
    PV(f"{prefix}Ampl:SumStd-I", auto_monitor=False) for prefix in REGBPM_PV_PREFIXES
]
REGBPM_COUNTS_PVS = [
    PV(f"{prefix}AmplV:Max-I", auto_monitor=False) for prefix in REGBPM_PV_PREFIXES
]

nRegBPM = len(REGBPM_PV_PREFIXES)

DCCT_PV = PV("SR:C03-BI{DCCT:1}I:Total-I", auto_monitor=False)

ABORT_PV_TIMEOUT = 0.1  # [s]
SUM_PVS_TIMEOUT = 0.3  # [s]
DCCT_PV_TIMEOUT = 0.1  # [s]

PV_STRS = json.loads(Path("pvs_lifetime.json").read_text())
PVS = {k: PV(pv_str, auto_monitor=False) for k, pv_str in PV_STRS.items()}

# Storage used by callbacks
CB_DATA = defaultdict(dict)
CB_ARGS = defaultdict(dict)


def callback_append_to_list(pvname, **kwargs):
    if False:
        debugpy.debug_this_thread()  # Needed for VSCode thread debugging
        print(kwargs)

    cb_data = CB_DATA[pvname]

    for k in ["value", "timestamp"]:
        if k not in cb_data:
            cb_data[k] = []
        cb_data[k].append(kwargs.get(k))


def _getSimulatedDCCT():
    raise NotImplementedError


def _getSimulatedSumSignals():
    raise NotImplementedError


def SimulatedCaFloat():
    raise NotImplementedError


def _get_nan_dict(timestamp, dcct_mA, meas_type="adaptive_lookback"):
    """"""

    filtered_avg_tau_hr = filtered_rms_tau_hr = np.nan
    raw_avg_tau_hr = raw_rms_tau_hr = np.nan
    raw_tau_hrs = np.zeros((nRegBPM,)) * np.nan
    bad_bpm_indexes = np.array([])
    outlier_indexes = np.array([])

    out = dict(
        dcct_mA=dcct_mA,
        timestamp=timestamp,
        raw_vec=raw_tau_hrs,
        avg=filtered_avg_tau_hr,
        rms=filtered_rms_tau_hr,
        raw_avg=raw_avg_tau_hr,
        raw_rms=raw_rms_tau_hr,
        bad_bpm_indexes=bad_bpm_indexes,
        outlier_indexes=outlier_indexes,
        no_beam=False,
    )

    if meas_type in ("adaptive", "adaptive_lookback"):
        out["nShots_used"] = 0
    if meas_type in ("adaptive_lookback",):
        out["avg_noise_array"] = np.zeros((nRegBPM,)) * np.nan

    return out


def measLifetimeAdaptivePeriod(
    max_wait=120.0,
    update_period=1.0,
    sigma_cut=3.0,
    sum_diff_thresh_fac=5.0,
    min_samples=5,
    abort_pv=None,
    mode="online",
    min_dcct_mA=0.2,
    measured_data=None,
    measured_data_time_index=None,
    show_plot=False,
    plt_show=True,
    profiler_on=False,
):
    """"""

    if mode not in ("online", "simulated", "measured"):
        raise ValueError(f'Invalid "mode": {mode}')
    if mode == "measured":
        if (measured_data is None) or (measured_data_time_index is None):
            raise ValueError(
                'If "mode" is "measured", then you must provide "measured_data" '
                'and "measured_data_time_index".'
            )
        else:
            MEASDATA = measured_data
    else:
        MEASDATA = None

    pvs = REGBPM_SUM_PVS[:]
    std_pvs = REGBPM_SUMSTD_PVS[:]

    N_bpm = nRegBPM

    bad_bpm_indexes = np.array([])

    moni_pv_keys = ["DCCT_1", "DCCT_2", "DCCT_precise", "eps_x_nm", "eps_y_pm"]
    moni_pv_keys += [_pv.pvname for _pv in REGBPM_SUM_PVS]
    for k in moni_pv_keys:
        add_callback(PVS[k], callback_append_to_list)
        turn_on_pv_monitor(PVS[k])
    for _pv in REGBPM_SUM_PVS + REGBPM_SUMSTD_PVS:
        add_callback(_pv, callback_append_to_list)
        turn_on_pv_monitor(_pv)

    N_samples = int(np.ceil(float(max_wait) / float(update_period)))

    t = np.zeros(N_samples) * np.nan
    I = np.zeros((N_samples, N_bpm)) * np.nan

    filled_index = None

    idle_time = 0.0
    loop_t0 = time.perf_counter()

    aborted = False

    for i in range(N_samples):
        t0 = time.perf_counter()

        if (mode == "online") and (abort_pv is not None):
            _aborted = abort_pv.get(timeout=ABORT_PV_TIMEOUT)
            if (_aborted is not None) and _aborted:
                aborted = True
                break
            else:
                aborted = False

        if mode == "online":
            for iFail in range(3):
                dcct_mA = DCCT_PV.get(timeout=DCCT_PV_TIMEOUT)
                if dcct_mA is not None:
                    break
                else:
                    time.sleep(1.0)
            else:
                raise RuntimeError("Too many failures in caget.")
        elif mode == "simulated":
            dcct_mA = _getSimulatedDCCT()
        else:
            try:
                dcct_mA = MEASDATA["dcct_mA"][measured_data_time_index]
                dcct_ts = (
                    MEASDATA["dcct_mA_t"][measured_data_time_index] + MEASDATA["t0"]
                )
            except IndexError:
                return None

        if dcct_mA < min_dcct_mA:
            if mode != "measured":
                out = _get_nan_dict(time.perf_counter(), dcct_mA, meas_type="adaptive")
            else:
                out = _get_nan_dict(dcct_ts, dcct_mA, meas_type="adaptive")
                measured_data_time_index += 1
                out["measured_data_time_index"] = measured_data_time_index
            out["no_beam"] = True
            return out
        else:
            for iFail in range(3):
                if mode == "online":
                    data = [pv.get(timeout=SUM_PVS_TIMEOUT) for pv in pvs + std_pvs]
                elif mode == "simulated":
                    data = _getSimulatedSumSignals(return_std=True)
                else:
                    data = [
                        SimulatedCaFloat(x, timestamp=ts)
                        for x, ts in zip(
                            np.append(
                                MEASDATA["sum_array"][measured_data_time_index],
                                MEASDATA["sumstd_array"][measured_data_time_index],
                            ),
                            np.append(
                                MEASDATA["sum_t_array"][measured_data_time_index],
                                MEASDATA["sumstd_t_array"][measured_data_time_index],
                            )
                            + MEASDATA["t0"],
                        )
                    ]
                    measured_data_time_index += 1
                sum_ok_indexes = [(d is not None) for d in data[:N_bpm]]
                std_ok_indexes = [(d is not None) for d in data[N_bpm:]]
                ok_indexes = np.logical_and(sum_ok_indexes, std_ok_indexes)
                if sum(ok_indexes) >= N_bpm * 0.80:
                    break
                else:
                    time.sleep(1.0)
            else:
                raise RuntimeError("Too many failures in caget.")

            assert len(pvs) == len(data[:N_bpm])
            timestamps = np.array(
                [
                    pv.timestamp if d is not None else np.nan
                    for pv, d in zip(pvs, data[:N_bpm])
                ]
            )
            nan_inds = np.isnan(timestamps)
            valid_ts_indexes = (
                np.abs(timestamps - np.median(timestamps[~nan_inds])) < update_period
            )
            valid_indexes = np.logical_and(ok_indexes, valid_ts_indexes)
            valid_sum_data = np.array([d.real for d in data[:N_bpm]])
            valid_sum_data[~valid_indexes] = np.nan
            noise_level = np.array([d.real for d in data[N_bpm:]])
            noise_level[~valid_indexes] = np.nan

        I[i, :] = valid_sum_data
        t[i] = np.mean(timestamps[~nan_inds])
        filled_index = i

        if i < 2:
            continue

        I_sub = I[:i, :]
        nan_col_inds = np.any(np.isnan(I_sub), axis=0)
        I_start = I_sub[0, ~nan_col_inds]
        I_end = I_sub[-1, ~nan_col_inds]
        all_out_of_noise = np.all(
            np.abs(I_start - I_end) > (noise_level[~nan_col_inds] * sum_diff_thresh_fac)
        )

        if all_out_of_noise and (i >= min_samples - 1):
            break

        if mode in ("online", "simulated"):
            if i + 1 < N_samples:
                idle = max([0.0, update_period - (time.perf_counter() - t0)])
                if profiler_on:
                    idle_time += idle
                time.sleep(idle)

    for k in moni_pv_keys:
        turn_off_pv_monitor(PVS[k])
    for _pv in REGBPM_SUM_PVS + REGBPM_SUMSTD_PVS:
        turn_off_pv_monitor(_pv)

    timestamp = time.perf_counter()
    if profiler_on:
        loop_t1 = time.perf_counter()
        print(f"Total while loop time [s] = {loop_t1-loop_t0:.3f}"),
        print("Total idling time [s] = {idle_time:.3f}")

    out = _get_nan_dict(t[filled_index], dcct_mA, meas_type="adaptive")
    if filled_index is not None:
        out["nShots_used"] = filled_index + 1
    if mode == "measured":
        out["measured_data_time_index"] = measured_data_time_index

    if filled_index is not None:
        I = I[: (filled_index + 1), :]
        t = t[: (filled_index + 1)]
        nan_inds = np.any(np.isnan(I), axis=0)
        if np.all(nan_inds):
            filtered_avg_tau_hr = filtered_rms_tau_hr = np.nan
            raw_avg_tau_hr = raw_rms_tau_hr = np.nan
            raw_tau_hrs = np.zeros((N_bpm,)) * np.nan
            outlier_indexes = np.array([])
        else:
            valid_bpm_indexes = [
                bi for bi in range(nRegBPM) if bi not in bad_bpm_indexes
            ]
            valid_bpm_indexes = np.array(valid_bpm_indexes)[~nan_inds]
            # Update bad bpm indexes
            bad_bpm_indexes = np.array(
                [bi for bi in range(nRegBPM) if bi not in valid_bpm_indexes]
            )
            nValidBPM = valid_bpm_indexes.size

            p1, p2 = np.polyfit(t - t[0], I[:, ~nan_inds], 1)
            raw_tau_hrs = -p2 / p1 / (60.0 * 60.0)

            median = np.median(raw_tau_hrs)
            sort_inds = np.argsort(np.abs(raw_tau_hrs - median))
            sorted_tau_hrs = raw_tau_hrs[sort_inds]
            valid_inds = np.array([True] * nValidBPM)
            outlier_count = 0
            for i in range(nValidBPM)[::-1]:
                if np.abs(sorted_tau_hrs[i] - median) <= np.std(
                    sorted_tau_hrs[:i] * sigma_cut
                ):
                    print(
                        f"** # of outliers found = {outlier_count} (out of {nValidBPM})"
                    )
                    break
                else:
                    outlier_count += 1
                    valid_inds[sort_inds[i]] = False
            filtered_avg_tau_hr = np.mean(raw_tau_hrs[valid_inds])
            filtered_rms_tau_hr = np.std(raw_tau_hrs[valid_inds], ddof=1)
            outlier_indexes = valid_bpm_indexes[~valid_inds]
            raw_avg_tau_hr = np.mean(raw_tau_hrs)
            raw_rms_tau_hr = np.std(raw_tau_hrs, ddof=1)
    else:
        filtered_avg_tau_hr = filtered_rms_tau_hr = np.nan
        raw_avg_tau_hr = raw_rms_tau_hr = np.nan
        raw_tau_hrs = np.zeros((N_bpm,)) * np.nan
        outlier_indexes = np.array([])

    if show_plot:
        plt.figure()
        plt.plot(raw_tau_hrs, ".")
        plt.xlabel("(Valid) BPM Index")
        plt.ylabel("Lifetime [hr]")
        plt.grid(True)
        plt.title(
            rf"Lifetime [hr] = {filtered_avg_tau_hr:.2f} $\pm$ {filtered_rms_tau_hr:.2f}"
        )

        if plt_show:
            plt.show()

    out["raw_vec"] = raw_tau_hrs
    out["avg"] = filtered_avg_tau_hr
    out["rms"] = filtered_rms_tau_hr
    out["raw_avg"] = raw_avg_tau_hr
    out["raw_rms"] = raw_rms_tau_hr
    out["bad_bpm_indexes"] = bad_bpm_indexes
    out["outlier_indexes"] = outlier_indexes
    out["aborted"] = aborted

    moni_data = {}
    for k in moni_pv_keys:
        pvname = PV_STRS[k]
        moni_data[k] = {k2: np.array(v2) for k2, v2 in CB_DATA[pvname].items()}
    for _pv in REGBPM_SUM_PVS:
        pvname = _pv.pvname
        cell_str, bpm_num_str = re.match("SR:(C\d\d)-BI\{BPM:(\d)\}", pvname).groups()
        k = f"sum_{cell_str}-P{bpm_num_str}"
        moni_data[k] = {k2: np.array(v2) for k2, v2 in CB_DATA[pvname].items()}
    for _pv in REGBPM_SUMSTD_PVS:
        pvname = _pv.pvname
        cell_str, bpm_num_str = re.match("SR:(C\d\d)-BI\{BPM:(\d)\}", pvname).groups()
        k = f"sumstd_{cell_str}-P{bpm_num_str}"
        moni_data[k] = {k2: np.array(v2) for k2, v2 in CB_DATA[pvname].items()}

    out["moni_data"] = moni_data

    return out


if __name__ == "__main__":
    out = measLifetimeAdaptivePeriod(
        max_wait=120.0,
        update_period=1.0,
        sigma_cut=3.0,
        sum_diff_thresh_fac=20.0,  # 10.0, #5.0,
        abort_pv=None,
        mode="online",
        min_dcct_mA=0.2,
        measured_data=None,
        measured_data_time_index=None,
        show_plot=False,
        plt_show=True,
        profiler_on=False,
    )
