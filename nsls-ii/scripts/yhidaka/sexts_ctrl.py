from pathlib import Path
import json
from collections import defaultdict

import numpy as np
import h5py
from scipy.interpolate import interp1d
from epics import PV

UNITCONV = {}
with h5py.File(Path("unitconv_grouped_sexts.h5"), "r") as f:
    for fam in list(f):
        UNITCONV[fam] = dict(
            phy=f[fam]["target_mean_ap_b2"][()], raw=f[fam]["current_A"][()]
        )

SEXT_PVSP_MAP = json.loads(Path("sext_pvsp_map.json").read_text())
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
    fam: [SEXT_GROUP2PV[SEXT_PVSP_MAP[f"{fam}_{i+1}"]["group"]]
           for i in range(n_kids)]
    for fam, n_kids in SEXT_FAM2NKIDS.items()
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
        family = SEXT_GROUP2FAM_IND[group]['family']
        index = SEXT_GROUP2FAM_IND[group]['index']
        pvs = [SEXT_FAM2PVS[family][index]]
    else:
        raise ValueError(f"No match found for 2nd argument: {family_or_group}")

    f = interp1d(
        UNITCONV[family]["phy"],
        UNITCONV[family]["raw"],
        kind="linear",
        axis=0,
        copy=True,
        bounds_error=True,
        fill_value=np.nan,
    )
    target_sp_raws = f(target_sp_phy)

    if group is None: # whole family
        outputs = target_sp_raws # [A]
    else:
        outputs = [target_sp_raws[index]]  # [A]

    assert len(pvs) == len(outputs)

    return pvs, outputs

def change_sext_strengths(target_sp_phy, family_or_group):

    """
    target_sp_phy [m^(-2)]
    """

    pvs, new_setpoints_A = convertGroupedSextPhySetpoints(target_sp_phy, family_or_group)

    assert len(pvs) == len(new_setpoints_A)
    for pv, amp in zip(pvs, new_setpoints_A):
        pv.put(amp)
        pv.get()


if __name__ == '__main__':

    base_K2L = 3.962 # [m^(-2)]
    change_sext_strengths(base_K2L, 'SH1')