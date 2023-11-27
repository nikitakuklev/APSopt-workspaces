from pathlib import Path
import json
from collections import defaultdict
import time

import numpy as np
import h5py
from scipy.interpolate import interp1d
from epics import PV

UNITCONV = {}
with h5py.File((Path(__file__).parent / "unitconv_grouped_sexts.h5"), "r") as f:
    for fam in list(f):
        UNITCONV[fam] = dict(
            phy=f[fam]["target_mean_ap_b2"][()], raw=f[fam]["current_A"][()]
        )

SEXT_PVSP_MAP = json.loads((Path(__file__).parent / "sext_pvsp_map.json").read_text())
_sext_fam_series_indexes = defaultdict(list)
for k in list(SEXT_PVSP_MAP):
    fam, series_num_str = k.split("_")
    _sext_fam_series_indexes[fam].append(int(series_num_str) - 1)
SEXT_FAM2NKIDS = {}
for fam, inds in _sext_fam_series_indexes.items():
    SEXT_FAM2NKIDS[fam] = len(inds)
    _sext_fam_series_indexes[fam] = sorted(inds)
    assert _sext_fam_series_indexes[fam] == list(range(len(inds)))
SEXT_GROUP2PV = {
    d["group"]: PV(d["pvname"], auto_monitor=False) for d in SEXT_PVSP_MAP.values()
}
SEXT_GROUP2FAM_IND = {
    d["group"]: dict(family=k.split("_")[0], index=int(k.split("_")[1]) - 1)
    for k, d in SEXT_PVSP_MAP.items()
}
SEXT_FAM2PVS = {
    fam: [SEXT_GROUP2PV[SEXT_PVSP_MAP[f"{fam}_{i+1}"]["group"]] for i in range(n_kids)]
    for fam, n_kids in SEXT_FAM2NKIDS.items()
}

SEXT_SPPVNAME2RBPVS = {
    pv.pvname: PV(pv.pvname.replace(":Sp1-SP", ":Ps1DCCT1-I"), auto_monitor=False)
    for pv in SEXT_GROUP2PV.values()
}


def convertGroupedSextPhySetpoints(target_sp_phy, family_or_group):
    """
    target_sp_phy [m^(-2)]
    outputs [A]
    """

    if family_or_group in SEXT_FAM2PVS:
        group = None
        family = family_or_group
        pvs = SEXT_FAM2PVS[family]
    elif family_or_group in SEXT_GROUP2FAM_IND:
        group = family_or_group
        family = SEXT_GROUP2FAM_IND[group]["family"]
        index = SEXT_GROUP2FAM_IND[group]["index"]
        pvs = [SEXT_FAM2PVS[family][index]]
    else:
        raise ValueError(f"No match found for 2nd argument: {family_or_group}")

    interp_func = interp1d(
        UNITCONV[family]["phy"],
        UNITCONV[family]["raw"],
        kind="linear",
        axis=0,
        copy=True,
        bounds_error=True,
        fill_value=np.nan,
    )
    target_sp_raws = interp_func(target_sp_phy)

    if group is None:  # whole family
        outputs = target_sp_raws  # [A]
    else:
        outputs = [target_sp_raws[index]]  # [A]

    assert len(pvs) == len(outputs)

    return pvs, outputs


# def change_sext_strengths(target_sp_phy, family_or_group, max_dI=1.0, step_wait=2.0,
#                           wait_small_SP_RB_diff=False, max_SP_RB_dI=0.05):
#     """
#     target_sp_phy [m^(-2)]
#     """

#     assert max_dI > 0.0
#     assert step_wait >= 0.0

#     if wait_small_SP_RB_diff:
#         assert max_SP_RB_dI > 0.0
#         max_wait = 30.0 # [s]

#     setpoint_pvs, new_setpoints_A = convertGroupedSextPhySetpoints(
#         target_sp_phy, family_or_group
#     )

#     new_setpoints_A = np.array(new_setpoints_A)
#     cur_setpoints_A = np.array([pv.get() for pv in setpoint_pvs])

#     diff_A = new_setpoints_A - cur_setpoints_A
#     nsteps = int(np.ceil(np.max(np.abs(diff_A)) / max_dI))

#     readback_pvs = [SEXT_SPPVNAME2RBPVS[_pv.pvname] for _pv in setpoint_pvs]

#     for iStep in range(nsteps):

#         new_As = cur_setpoints_A + diff_A * (iStep + 1) / nsteps

#         for _pv, amp in zip(setpoint_pvs, new_As):
#             _pv.put(amp)
#         for _pv in setpoint_pvs:
#             _pv.get()

#         if iStep != nsteps - 1:
#             if not wait_small_SP_RB_diff:
#                 time.sleep(step_wait)
#             else:
#                 t_wait_start = time.perf_counter()
#                 while time.perf_counter() - t_wait_start < max_wait:
#                     cur_As = np.array([_pv.get() for _pv in readback_pvs])
#                     diff_As = new_As - cur_As
#                     if np.max(np.abs(diff_As)) < max_SP_RB_dI:
#                         break
#                     else:
#                         time.sleep(0.3)


def change_sext_strengths(
    target_sp_phy_list, family_or_group_list, max_dI=1.0, step_wait=2.0
):
    """
    target_sp_phy [m^(-2)]
    """

    assert max_dI > 0.0
    assert step_wait >= 0.0

    new_setpoints_A_list = []
    cur_setpoints_A_list = []
    setpoint_pvs_list = []

    for target_sp_phy, family_or_group in zip(target_sp_phy_list, family_or_group_list):
        setpoint_pvs, new_setpoints_A = convertGroupedSextPhySetpoints(
            target_sp_phy, family_or_group
        )

        new_setpoints_A_list.extend(new_setpoints_A)
        cur_setpoints_A_list.extend([pv.get() for pv in setpoint_pvs])
        setpoint_pvs_list.extend(setpoint_pvs)

    new_setpoints_A = np.array(new_setpoints_A_list)
    cur_setpoints_A = np.array(cur_setpoints_A_list)

    diff_A = new_setpoints_A - cur_setpoints_A
    nsteps = int(np.ceil(np.max(np.abs(diff_A)) / max_dI))

    for iStep in range(nsteps):
        new_As = cur_setpoints_A + diff_A * (iStep + 1) / nsteps

        for _pv, amp in zip(setpoint_pvs_list, new_As):
            _pv.put(amp)
        for _pv in setpoint_pvs_list:
            _pv.get()

        if iStep != nsteps - 1:
            time.sleep(step_wait)


if __name__ == "__main__":
    base_K2L = {}
    base_K2L["SH1"] = 3.9666635820091023  # [m^(-2)]
    base_K2L["SM1A"] = -4.7793535935570315
    base_K2L["SM1B"] = -5.224932910164227
    base_K2L["SM2B"] = 7.164374888638177

    L = {}
    L["SM1A"] = 0.2
    L["SM1B"] = 0.2
    L["SM2B"] = 0.25

    if False:
        family = "SH1"
        new_K2L = base_K2L[family] * 1.0
        change_sext_strengths(new_K2L, family)

    if False:
        family = "SM1A"
        new_K2L = base_K2L[family] * 1.0
        change_sext_strengths(new_K2L, family)
    else:
        null_K2s = np.array([-0.6637, 0.7448, -0.0692])
        scaling = 0.0  # 5.0
        for family, null_dK2 in zip(["SM1A", "SM1B", "SM2B"], null_K2s):
            new_K2L = base_K2L[family] + null_dK2 * L[family] * scaling
