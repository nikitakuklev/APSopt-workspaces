from datetime import datetime
from pathlib import Path
import json
import time

import numpy as np

USE_PYEPICS = True
if not USE_PYEPICS:
    from cothread.catools import caget#, caput
    caput = print
else:
    from epics import PV

PV_STRS = {
    'BBFB_X_SP': "IGPF:TransFBX:FBCTRL",
    'BBFB_Y_SP': "IGPF:TransFBY:FBCTRL",
    'BBFB_X_RB': 'SR:OPS-BI{IGPF}FBX:Ctrl-Sts',
    'BBFB_Y_RB': 'SR:OPS-BI{IGPF}FBY:Ctrl-Sts',
    'Y_Feedback_pattern': 'IGPF:TransFBY:FB:PATTERN',
    'Y_Drive_pattern': 'IGPF:TransFBY:DRIVE:PATTERN',
    'Y_Drive_amplitude': 'IGPF:TransFBY:DRIVE:AMPL',
    'Y_Drive_freq': 'IGPF:TransFBY:DRIVE:FREQ',
    'Y_Drive_period': 'IGPF:TransFBY:DRIVE:PERIOD',
    'Y_SRAM_trig': 'IGPF:TransFBY:SRAM:HWTEN', # 0 := SOFT; 1 := HARD
    'RF_freq_RB': 'RF{FCnt:1}Freq:I',
}
if USE_PYEPICS:
    PVS = {k: PV(pv_str, auto_monitor=False) for k, pv_str in PV_STRS.items()}
else:
    PVS = PV_STRS

OPS = json.loads(Path('20231108T114817_BxB_ops_conditions.json').read_text())


def capture_ops_conditions():

    d = {}
    for k, pv in PVS.items():
        d[k] = dict(pv=pv.pvname, val=pv.get())

    output_filepath = Path(f'{datetime.now():%Y%m%dT%H%M%S}_BxB_ops_conditions.json')
    output_filepath.write_text(json.dumps(d, indent=2))

def conv_nu_to_kHz(nu, rf_freq_Hz=None):

    frac_nu = nu - np.floor(nu)

    if not rf_freq_Hz:
        rf_freq_Hz = caget('RF{FCnt:1}Freq:I')
    h = 1320

    fb_freq_Hz = rf_freq_Hz / h * frac_nu

    return fb_freq_Hz / 1e3

def conv_kHz_to_nu(nu_kHz, rf_freq_Hz=None):

    nu_Hz = nu_kHz * 1e3

    if not rf_freq_Hz:
        rf_freq_Hz = caget('RF{FCnt:1}Freq:I')
    h = 1320

    frac_nu = nu_Hz / (rf_freq_Hz / h)

    return frac_nu

def get_grow_freq_kHz(nu, rf_freq_Hz=None):

    frac_nu = nu - np.floor(nu)

    return conv_nu_to_kHz(1 - frac_nu, rf_freq_Hz=rf_freq_Hz)

def get_dampen_freq_kHz(nu, rf_freq_Hz=None):

    frac_nu = nu - np.floor(nu)

    return conv_nu_to_kHz(frac_nu, rf_freq_Hz=rf_freq_Hz)

def _change_BxB_feedback_state(plane, on):

    if on:
        non_desired = 0
        desired = 1
    else:
        non_desired = 1
        desired = 0

    Plane = plane.upper()
    if caget(OPS[f'BBFB_{Plane}_RB']['pv']) == non_desired:
        caput(OPS[f'BBFB_{Plane}_SP']['pv'], desired, wait=True)
        time.sleep(1.0)
        assert caget(OPS[f'BBFB_{Plane}_RB']['pv']) == desired

def turn_on_BxB_feedback():
    _change_BxB_feedback_state('x', True)
    _change_BxB_feedback_state('y', True)

def turn_off_BxB_feedback():
    _change_BxB_feedback_state('x', False)
    _change_BxB_feedback_state('y', False)

def turn_on_BxB_x_feedback():

    _change_BxB_feedback_state('x', True)

def turn_off_BxB_x_feedback():

    _change_BxB_feedback_state('x', False)

def turn_on_BxB_y_feedback():

    _change_BxB_feedback_state('y', True)

def turn_off_BxB_y_feedback():

    _change_BxB_feedback_state('y', False)

def get_ops_nuy():

    ops_nuy = caget('SR:OPS-HLA{TuneCorr}nuY-target')

    return ops_nuy

def restore_ops_settings(ops_nuy=None):

    pv_keys = [
        'Y_Feedback_pattern', 'Y_Drive_pattern', 'Y_Drive_amplitude',
        'Y_Drive_freq', 'Y_Drive_period', 'Y_SRAM_trig']
    pv_list = [OPS[k]['pv'] for k in pv_keys]
    ops_vals = [int(OPS[k]['val']) if k in ('Y_Drive_freq', 'Y_Drive_period')
                else OPS[k]['val'] for k in pv_keys]

    # Make sure the vertical BxB feedback is turned off first
    turn_off_BxB_y_feedback()

    # Override the dampening frequency with the current target nuy
    if not ops_nuy:
        ops_nuy = get_ops_nuy()
    ops_nuy = ops_nuy - np.floor(ops_nuy)
    drive_freq_kHz = int(np.round(get_dampen_freq_kHz(ops_nuy)))
    ops_vals[pv_keys.index('Y_Drive_freq')] = drive_freq_kHz

    caput_many(pv_list, ops_vals, wait=True)
    caget_many(pv_list)

    turn_on_BxB_y_feedback()

def clean_bunches(clean_ini_bunch_number, clean_end_bunch_number,
                  keep_ini_bunch_number, keep_end_bunch_number,
                  nuy):
    """
    To kill bunches 1 to 100, and keep 101 through 1250 for nuy = 0.27:

    clean_bunches(1, 100, 101, 1250, 0.27)
    """

    bunch_states = np.zeros(1320).astype(int)

    assert clean_ini_bunch_number <= clean_end_bunch_number
    clean_s_ = np.s_[(clean_ini_bunch_number-1):clean_end_bunch_number]
    assert np.all(bunch_states[clean_s_] == 0)
    bunch_states[clean_s_] = 1

    assert keep_ini_bunch_number <= keep_end_bunch_number
    keep_s_ = np.s_[(keep_ini_bunch_number-1):keep_end_bunch_number]
    assert np.all(bunch_states[keep_s_] == 0)
    bunch_states[clean_s_] = 2

    clean_pattern = f'{clean_ini_bunch_number}:{clean_end_bunch_number}'
    keep_pattern = f'{keep_ini_bunch_number}:{keep_end_bunch_number}'

    clean_ampl = 1.0
    clean_freq_kHz = int(np.round(get_grow_freq_kHz(nuy)))
    clean_period = 1_000_000
    clean_trig = 0 # 0 for SOFT

    pv_keys = [
        'Y_Feedback_pattern', 'Y_Drive_pattern', 'Y_Drive_amplitude',
        'Y_Drive_freq', 'Y_Drive_period', 'Y_SRAM_trig']
    pv_vals = [
        keep_pattern, clean_pattern, clean_ampl,
        clean_freq_kHz, clean_period, clean_trig
    ]
    pv_list = [PVS[k] for k in pv_keys]

    turn_on_BxB_y_feedback()

    caput_many(pv_list, pv_vals, wait=True)
    caget_many(pv_list)

    time.sleep(5.0)


if __name__ == '__main__':

    if False:
        capture_ops_conditions()



    if False: # TO-BE-DELETED
        import numpy as np
        import matplotlib.pyplot as plt

        csv_2d_filename = "csv-8.csv"
        csv_yproj_filename = "vprojection.csv"

        first_line = np.loadtxt(csv_2d_filename, skiprows=0, max_rows=1, dtype=object, delimiter=',')
        header_tokens = []
        first_row = []
        for s in first_line:
            try:
                v = float(s)
                first_row.append(v)
            except:
                header_tokens.append(s)
        header = ','.join(header_tokens)
        y_pos = np.array(first_row)
        rest = np.loadtxt(csv_2d_filename, skiprows=1, delimiter=',')
        x_pos = rest[:, 0]
        intensity = rest[:, 1:]

        plt.figure()
        plt.pcolor(y_pos * 1e3, x_pos * 1e3, intensity, cmap='jet')
        plt.ylabel('x [mm]')
        plt.xlabel('y [mm]')
        plt.colorbar()

        yproj = np.sum(intensity, axis=0)
        xproj = np.sum(intensity, axis=1)

        vproj_from_csv = np.loadtxt(csv_yproj_filename, skiprows=1, delimiter=',')

        plt.figure()
        plt.plot(y_pos * 1e3, yproj / np.max(yproj), 'r.-', label='From 2D array csv')
        plt.plot(vproj_from_csv[:, 0] * 1e3, vproj_from_csv[:, 1] / np.max(vproj_from_csv[:, 1]), 'b.-', label='From projection csv')
        plt.legend(loc='best')
        plt.xlabel('y [mm]')
        plt.ylabel('Normalized projected intensity')

        plt.figure()
        plt.plot(x_pos * 1e3, xproj, '.-')
        plt.xlabel('x [mm]')

        plt.show()
