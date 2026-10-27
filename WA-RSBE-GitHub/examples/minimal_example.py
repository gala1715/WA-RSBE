import numpy as np
from wa_rsbe import ObserverConfig, WarpAwareSuspensionObserver

cfg = ObserverConfig()
obs = WarpAwareSuspensionObserver(cfg)

stationary = {
    'ax_specific_mps2': 0.0,
    'ay_specific_mps2': 0.0,
    'az_specific_mps2': cfg.g,
    'roll_rate_p_radps': 0.0,
    'pitch_rate_q_radps': 0.0,
    'yaw_rate_r_radps': 0.0,
    'suspension_disp_FL_m': 0.0,
    'suspension_disp_FR_m': 0.0,
    'suspension_disp_RL_m': 0.0,
    'suspension_disp_RR_m': 0.0,
}
obs.initialize_from_warmup([stationary] * 100)
y = obs.step(stationary)
print('road_pitch_deg =', np.degrees(y['road_pitch_rad']))
print('road_bank_deg  =', np.degrees(y['road_bank_rad']))
print('planar_conf    =', y['planar_confidence'])
print('gravity_conf   =', y['gravity_confidence'])
