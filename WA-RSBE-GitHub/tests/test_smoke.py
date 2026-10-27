import math
from wa_rsbe import ObserverConfig, WarpAwareSuspensionObserver

def test_stationary_smoke():
    cfg = ObserverConfig()
    obs = WarpAwareSuspensionObserver(cfg)
    row = {
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
    obs.initialize_from_warmup([row] * 100)
    y = obs.step(row)
    assert math.isfinite(y['road_pitch_rad'])
    assert math.isfinite(y['road_bank_rad'])
    assert abs(y['road_pitch_rad']) < 1e-6
    assert abs(y['road_bank_rad']) < 1e-6
