\
    # WA-RSBE algorithm notes

    The online input contract is deliberately limited to a six-axis IMU and four
    suspension displacement sensors:

    - `ax_specific_mps2`, `ay_specific_mps2`, `az_specific_mps2`
    - `roll_rate_p_radps`, `pitch_rate_q_radps`, `yaw_rate_r_radps`
    - `suspension_disp_FL_m`, `suspension_disp_FR_m`,
      `suspension_disp_RL_m`, `suspension_disp_RR_m`

    The four suspension measurements are decomposed into common/heave, front-rear,
    left-right and diagonal warp modes. The diagonal mode is used as a spatial
    non-coplanarity indicator. The estimator combines:

    1. suspension-derived relative pitch/bank geometry;
    2. low-frequency IMU gravity attitude;
    3. gyroscope + suspension-angle-rate observations;
    4. road-planarity confidence and gravity confidence;
    5. a 4-state constant-rate Kalman observer;
    6. bounded absolute-angle innovations.

    The state is

    `x = [road_pitch, road_pitch_rate, road_bank, road_bank_rate]^T`.

    Key implementation constants in the current V5 snapshot include:

    - sample period: 0.01 s (100 Hz)
    - wheelbase: 3.000 m
    - track width: 1.710 m
    - absolute-angle innovation limit: 4 deg
    - minimum slope-rate confidence: 0.02
    - minimum bank-rate confidence: 0.40

    ## Important implementation-alignment note

    The archived V5 code also contains causal large-inclination pitch-geometry and
    coupled bank-geometry compensation switches. They are enabled by default in
    the exact code snapshot used for the included O27R1 reproduction. If this
    repository is released with a manuscript, verify that every enabled
    implementation detail is described consistently in the paper.
