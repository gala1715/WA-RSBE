
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Mapping, Optional

import numpy as np


@dataclass
class ObserverConfig:
    # Sampling / geometry
    dt: float = 0.01
    g: float = 9.80665
    wheelbase_m: float = 3.000
    track_m: float = 1.710

    # Physical suspension-sensor -> wheel vertical motion ratio.
    # This MUST come from suspension geometry / bench calibration, not Y truth.
    suspension_motion_ratio: float = 1.0

    # Dataset sign convention:
    # positive road pitch ~= -q integration; positive bank ~= +p integration
    pitch_gyro_sign: float = -1.0
    roll_gyro_sign: float = +1.0

    # Causal conditioning
    accel_tilt_tau_s: float = 0.35
    modal_rate_tau_s: float = 0.05
    accel_norm_rate_tau_s: float = 0.05

    # Robust startup / warp references
    min_warp_sigma_m: float = 0.001
    min_warp_rate_sigma_mps: float = 0.02
    warp_ref_floor_m: float = 0.003
    warp_rate_ref_floor_mps: float = 0.05

    # Gravity-anchor confidence.
    # No Vx is used. Dynamic acceleration is handled by confidence reduction.
    accel_norm_ref_mps2: float = 0.35
    accel_norm_rate_ref_mps3: float = 3.0
    yaw_rate_ref_radps: float = 0.25
    bank_yaw_rate_ref_radps: float = 0.15

    # KF noise
    abs_angle_sigma_deg: float = 0.35
    rate_sigma_degps: float = 1.5
    terrain_ang_acc_sigma_degps2: float = 8.0
    innovation_clip_deg: float = 4.0

    # Large-inclination suspension-geometry compensation (LIGC).
    # This is inactive around small body inclinations and progressively
    # strengthens the suspension relative-pitch correction on steep grades.
    large_inclination_compensation: bool = True
    large_inclination_threshold_deg: float = 10.0
    large_inclination_gain: float = 6.0
    large_inclination_gain_max: float = 4.0
    large_inclination_bias_tau_s: float = 3.0

    # Coupled large-inclination bank compensation (CLBC).
    # Activated only when a large longitudinal inclination and a sufficiently
    # large bank occur simultaneously. All gating variables are online IMU
    # quantities; no truth, Vx, wheel speed, or tire model is used online.
    large_bank_compensation: bool = True
    bank_comp_threshold_deg: float = 6.0
    bank_comp_full_deg: float = 10.0
    bank_geometry_gain_max: float = 3.7
    bank_comp_tau_s: float = 0.9
    bank_pitch_gate_threshold_deg: float = 24.0
    bank_pitch_gate_full_deg: float = 30.0

    # confidence lower bounds
    min_abs_confidence: float = 0.01
    min_rate_confidence: float = 0.02
    min_bank_rate_confidence: float = 0.40


class FirstOrderLPF:
    """Causal first-order low-pass filter; no future samples."""
    def __init__(self, dt: float, tau: float):
        self.alpha = 1.0 if tau <= 0 else dt / (tau + dt)
        self.y: Optional[float] = None

    def reset(self) -> None:
        self.y = None

    def update(self, x: float) -> float:
        x = float(x)
        if self.y is None:
            self.y = x
        else:
            self.y += self.alpha * (x - self.y)
        return self.y


def robust_median_sigma(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return med, max(1.4826 * mad, 1e-9)


class WarpAwareSuspensionObserver:
    r"""
    No-Vx, 100-Hz road pitch/bank observer with slowly varying
    large-inclination suspension-geometry compensation (WA-RSBE optimized).

    Core online inputs:
      ax, ay, az, p, q, r, four suspension displacements.

    It intentionally does NOT consume:
      wheel speed, Vx, Vy, drive torque, brake torque, Y truth,
      event_code, privileged channels, or future samples.

    Method
    ------
    1) The four suspension sensors are decomposed into:
         H = (FL+FR+RL+RR)/4
         P = (FL+FR-RL-RR)/4
         R = (FL-FR+RL-RR)/4
         W = (FL-FR-RL+RR)/4
       W is the non-planar / articulation mode.

    2) Suspension P/R form relative body-to-support-plane angles:
         alpha_s = atan2(2P/mr, L)
         beta_s  = atan2(2R/mr, T)

    3) No-Vx absolute road-angle pseudo-measurements:
         theta_abs = theta_gravity_body + alpha_s
         phi_abs   = phi_gravity_body   + beta_s

       theta_gravity_body / phi_gravity_body are low-frequency gravity
       directions from IMU. They are NOT trusted under strong dynamics.

    4) No-Vx road-angle-rate pseudo-measurements:
         theta_dot = sign_q*(q-bq) + d(alpha_s)/dt
         phi_dot   = sign_p*(p-bp) + d(beta_s)/dt

       Thus road-angle rate comes from body angular rate + measured relative
       suspension-plane rate, rather than vehicle longitudinal speed.

    5) Warp-aware covariance:
       when |W| or |dW/dt| grows, the current four-wheel support is likely
       non-coplanar, so both suspension geometry and body angular transients
       are down-weighted. The macro road orientation is prevented from
       following a pothole/rut/rock as if it were a true slope change.

    State:
       x = [road_pitch, road_pitch_rate, road_bank, road_bank_rate]^T
    """

    REQUIRED = (
        "ax_specific_mps2",
        "ay_specific_mps2",
        "az_specific_mps2",
        "roll_rate_p_radps",
        "pitch_rate_q_radps",
        "yaw_rate_r_radps",
        "suspension_disp_FL_m",
        "suspension_disp_FR_m",
        "suspension_disp_RL_m",
        "suspension_disp_RR_m",
    )

    def __init__(self, cfg: ObserverConfig = ObserverConfig(), mode: str = "full"):
        if mode not in {"full", "suspension_no_warp", "imu_only"}:
            raise ValueError("mode must be: full | suspension_no_warp | imu_only")
        self.cfg = cfg
        self.mode = mode
        self._build_filters()
        self.reset()

    def _build_filters(self) -> None:
        c = self.cfg
        self.theta_acc_lpf = FirstOrderLPF(c.dt, c.accel_tilt_tau_s)
        self.phi_acc_lpf = FirstOrderLPF(c.dt, c.accel_tilt_tau_s)
        self.alpha_rate_lpf = FirstOrderLPF(c.dt, c.modal_rate_tau_s)
        self.alpha_eff_rate_lpf = FirstOrderLPF(c.dt, c.modal_rate_tau_s)
        self.alpha_comp_lpf = FirstOrderLPF(c.dt, c.large_inclination_bias_tau_s)
        self.beta_rate_lpf = FirstOrderLPF(c.dt, c.modal_rate_tau_s)
        self.beta_eff_rate_lpf = FirstOrderLPF(c.dt, c.modal_rate_tau_s)
        self.beta_comp_lpf = FirstOrderLPF(c.dt, c.bank_comp_tau_s)
        self.warp_rate_lpf = FirstOrderLPF(c.dt, c.modal_rate_tau_s)
        self.anorm_rate_lpf = FirstOrderLPF(c.dt, c.accel_norm_rate_tau_s)

    def reset(self) -> None:
        self._build_filters()
        self.x = np.zeros(4, dtype=float)
        self.P = np.diag([
            math.radians(2.0) ** 2,
            math.radians(5.0) ** 2,
            math.radians(2.0) ** 2,
            math.radians(5.0) ** 2,
        ])
        self.susp_zero = np.zeros(4, dtype=float)
        self.gyro_bias_q = 0.0
        self.gyro_bias_p = 0.0
        self.warp_zero = 0.0
        self.warp_sigma = self.cfg.min_warp_sigma_m
        self.warp_rate_sigma = self.cfg.min_warp_rate_sigma_mps
        self.prev_alpha: Optional[float] = None
        self.prev_alpha_eff: Optional[float] = None
        self.prev_beta: Optional[float] = None
        self.prev_beta_eff: Optional[float] = None
        self.prev_warp: Optional[float] = None
        self.prev_anorm: Optional[float] = None
        self.initialized = False

    @staticmethod
    def modal(s: np.ndarray) -> tuple[float, float, float, float]:
        fl, fr, rl, rr = [float(v) for v in s]
        H = (fl + fr + rl + rr) / 4.0
        P = (fl + fr - rl - rr) / 4.0
        R = (fl - fr + rl - rr) / 4.0
        W = (fl - fr - rl + rr) / 4.0
        return H, P, R, W

    def _pitch_geometry_gain(self, theta_body_gravity: float) -> float:
        """Inclination-aware effective motion-ratio correction for road pitch.

        The correction is unity in the small-angle region. Above the configured
        threshold, its target magnitude grows smoothly with tan^2 of the excess
        inclination and is bounded to avoid excessive amplification. The added
        correction is applied through a slow causal filter. Only online IMU attitude
        is used; no road-angle truth, Vx, wheel speed, or tire model is used.
        """
        c = self.cfg
        if not c.large_inclination_compensation:
            return 1.0
        th = math.radians(c.large_inclination_threshold_deg)
        excess = max(abs(float(theta_body_gravity)) - th, 0.0)
        gain = 1.0 + c.large_inclination_gain * math.tan(excess) ** 2
        return min(gain, c.large_inclination_gain_max)

    def _bank_geometry_gain(self, phi_body_gravity: float,
                            theta_body_gravity: float = 0.0) -> float:
        c = self.cfg
        if not c.large_bank_compensation:
            return 1.0

        b0 = math.radians(c.bank_comp_threshold_deg)
        b1 = math.radians(max(c.bank_comp_full_deg,
                              c.bank_comp_threshold_deg + 1e-6))
        ub = (abs(float(phi_body_gravity)) - b0) / max(b1 - b0, 1e-9)
        ub = min(max(ub, 0.0), 1.0)
        sb = ub * ub * (3.0 - 2.0 * ub)

        p0 = math.radians(c.bank_pitch_gate_threshold_deg)
        p1 = math.radians(max(c.bank_pitch_gate_full_deg,
                              c.bank_pitch_gate_threshold_deg + 1e-6))
        up = (abs(float(theta_body_gravity)) - p0) / max(p1 - p0, 1e-9)
        up = min(max(up, 0.0), 1.0)
        sp = up * up * (3.0 - 2.0 * up)

        activation = sb * sp
        return 1.0 + (c.bank_geometry_gain_max - 1.0) * activation

    def initialize_from_warmup(self, warmup_rows: Iterable[Mapping[str, float]]) -> None:
        """
        Startup calibration only from online sensor channels.

        No Y truth is used:
          - suspension sensor zeros
          - gyro q/p biases
          - startup warp zero and robust warp noise scale
          - initial road angles from the same no-Vx pseudo-measurements
        """
        rows = list(warmup_rows)
        if not rows:
            raise ValueError("warmup_rows must not be empty")

        susp = np.array([
            [
                float(r["suspension_disp_FL_m"]),
                float(r["suspension_disp_FR_m"]),
                float(r["suspension_disp_RL_m"]),
                float(r["suspension_disp_RR_m"]),
            ]
            for r in rows
        ])
        self.susp_zero = np.median(susp, axis=0)

        q = np.array([float(r["pitch_rate_q_radps"]) for r in rows])
        p = np.array([float(r["roll_rate_p_radps"]) for r in rows])
        self.gyro_bias_q = float(np.median(q))
        self.gyro_bias_p = float(np.median(p))

        warp_raw = np.array([self.modal(s - self.susp_zero)[3] for s in susp])
        self.warp_zero, sigma_w = robust_median_sigma(warp_raw)
        self.warp_sigma = max(sigma_w, self.cfg.min_warp_sigma_m)

        if len(warp_raw) > 1:
            dw = np.diff(warp_raw) / self.cfg.dt
            _, sigma_dw = robust_median_sigma(dw)
        else:
            sigma_dw = self.cfg.min_warp_rate_sigma_mps
        self.warp_rate_sigma = max(sigma_dw, self.cfg.min_warp_rate_sigma_mps)

        # Estimate startup angle only from current sensors.
        theta0 = []
        phi0 = []
        mr = max(self.cfg.suspension_motion_ratio, 1e-6)
        for row, s in zip(rows, susp):
            _, Pm, Rm, _ = self.modal(s - self.susp_zero)
            alpha_s = math.atan2(2.0 * Pm / mr, self.cfg.wheelbase_m)
            beta_s = math.atan2(2.0 * Rm / mr, self.cfg.track_m)

            ax = float(row["ax_specific_mps2"])
            ay = float(row["ay_specific_mps2"])
            az = float(row["az_specific_mps2"])

            theta_b = math.atan2(ax, math.sqrt(ay * ay + az * az))
            phi_b = math.atan2(ay, az)
            pitch_gain = self._pitch_geometry_gain(theta_b)
            alpha_eff = alpha_s * pitch_gain
            bank_gain = self._bank_geometry_gain(phi_b, theta_b)
            beta_eff = beta_s * bank_gain

            if self.mode == "imu_only":
                theta0.append(theta_b)
                phi0.append(phi_b)
            else:
                theta0.append(theta_b + alpha_eff)
                phi0.append(phi_b + beta_eff)

        self.x[:] = [float(np.median(theta0)), 0.0, float(np.median(phi0)), 0.0]
        self.initialized = True

    def step(self, sig: Mapping[str, float]) -> Dict[str, float]:
        if not self.initialized:
            raise RuntimeError("Call initialize_from_warmup() before step().")

        c = self.cfg
        dt = c.dt
        g = c.g
        mr = max(c.suspension_motion_ratio, 1e-6)

        ax = float(sig["ax_specific_mps2"])
        ay = float(sig["ay_specific_mps2"])
        az = float(sig["az_specific_mps2"])
        p = float(sig["roll_rate_p_radps"])
        q = float(sig["pitch_rate_q_radps"])
        r = float(sig["yaw_rate_r_radps"])

        susp = np.array([
            sig["suspension_disp_FL_m"],
            sig["suspension_disp_FR_m"],
            sig["suspension_disp_RL_m"],
            sig["suspension_disp_RR_m"],
        ], dtype=float) - self.susp_zero

        _, Pm, Rm, Wm_raw = self.modal(susp)
        Wm = Wm_raw - self.warp_zero

        # Suspension-derived relative body/support-plane angles.
        alpha_s = math.atan2(2.0 * Pm / mr, c.wheelbase_m)
        beta_s = math.atan2(2.0 * Rm / mr, c.track_m)

        if self.prev_alpha is None:
            alpha_rate_raw = 0.0
            beta_rate_raw = 0.0
            warp_rate_raw = 0.0
        else:
            alpha_rate_raw = (alpha_s - self.prev_alpha) / dt
            beta_rate_raw = (beta_s - self.prev_beta) / dt
            warp_rate_raw = (Wm - self.prev_warp) / dt

        self.prev_alpha = alpha_s
        self.prev_beta = beta_s
        self.prev_warp = Wm

        alpha_rate = self.alpha_rate_lpf.update(alpha_rate_raw)
        beta_rate = self.beta_rate_lpf.update(beta_rate_raw)
        warp_rate = self.warp_rate_lpf.update(warp_rate_raw)

        # Body gravity direction, no Vx.
        theta_body_gravity = self.theta_acc_lpf.update(
            math.atan2(ax, math.sqrt(ay * ay + az * az))
        )
        phi_body_gravity = self.phi_acc_lpf.update(
            math.atan2(ay, az)
        )

        # Coupled large-inclination bank compensation.
        # The added bank correction is slowly filtered so that the macro-bank
        # channel does not follow wheel-level roughness or high-frequency shock.
        bank_geometry_gain = self._bank_geometry_gain(
            phi_body_gravity, theta_body_gravity
        )
        beta_comp_raw = (bank_geometry_gain - 1.0) * beta_s
        beta_comp = self.beta_comp_lpf.update(beta_comp_raw)
        beta_eff = beta_s + beta_comp

        if self.prev_beta_eff is None:
            beta_eff_rate_raw = 0.0
        else:
            beta_eff_rate_raw = (beta_eff - self.prev_beta_eff) / dt
        self.prev_beta_eff = beta_eff
        beta_eff_rate = self.beta_eff_rate_lpf.update(beta_eff_rate_raw)

        # Large-inclination geometry correction. The original fast suspension
        # pitch is retained. Only the additional large-angle correction is
        # low-pass filtered, so steep-grade bias is corrected without
        # amplifying rough-road suspension oscillations.
        pitch_geometry_gain = self._pitch_geometry_gain(theta_body_gravity)
        alpha_comp_raw = (pitch_geometry_gain - 1.0) * alpha_s
        alpha_comp = self.alpha_comp_lpf.update(alpha_comp_raw)
        alpha_eff = alpha_s + alpha_comp
        if self.prev_alpha_eff is None:
            alpha_eff_rate_raw = 0.0
        else:
            alpha_eff_rate_raw = (alpha_eff - self.prev_alpha_eff) / dt
        self.prev_alpha_eff = alpha_eff
        alpha_eff_rate = self.alpha_eff_rate_lpf.update(alpha_eff_rate_raw)

        if self.mode == "imu_only":
            theta_abs = theta_body_gravity
            phi_abs = phi_body_gravity
            theta_rate = c.pitch_gyro_sign * (q - self.gyro_bias_q)
            phi_rate = c.roll_gyro_sign * (p - self.gyro_bias_p)
        else:
            theta_abs = theta_body_gravity + alpha_eff
            phi_abs = phi_body_gravity + beta_eff
            theta_rate = (
                c.pitch_gyro_sign * (q - self.gyro_bias_q)
                + alpha_eff_rate
            )
            phi_rate = (
                c.roll_gyro_sign * (p - self.gyro_bias_p)
                + beta_eff_rate
            )

        # ---------- Warp / non-planarity confidence ----------
        if self.mode == "full":
            warp_ref = max(3.0 * self.warp_sigma, c.warp_ref_floor_m)
            warp_rate_ref = max(
                3.0 * self.warp_rate_sigma,
                c.warp_rate_ref_floor_mps,
            )
            z_w = abs(Wm) / warp_ref
            z_dw = abs(warp_rate) / warp_rate_ref
            planar_conf = math.exp(-0.5 * (z_w * z_w + 0.15 * z_dw * z_dw))
        else:
            planar_conf = 1.0

        # ---------- No-Vx gravity-anchor confidence ----------
        anorm = math.sqrt(ax * ax + ay * ay + az * az)
        if self.prev_anorm is None:
            anorm_rate_raw = 0.0
        else:
            anorm_rate_raw = (anorm - self.prev_anorm) / dt
        self.prev_anorm = anorm
        anorm_rate = self.anorm_rate_lpf.update(anorm_rate_raw)

        dyn_score = (
            (abs(anorm - g) / c.accel_norm_ref_mps2) ** 2
            + 0.15 * (abs(anorm_rate) / c.accel_norm_rate_ref_mps3) ** 2
            + 0.40 * (abs(r) / c.yaw_rate_ref_radps) ** 2
        )
        gravity_conf = math.exp(-0.5 * min(dyn_score, 50.0))

        abs_conf_pitch = max(
            c.min_abs_confidence,
            planar_conf * gravity_conf,
        )
        abs_conf_bank = max(
            c.min_abs_confidence,
            planar_conf
            * gravity_conf
            * math.exp(-0.5 * (abs(r) / c.bank_yaw_rate_ref_radps) ** 2),
        )
        rate_conf_pitch = max(c.min_rate_confidence, planar_conf)
        rate_conf_bank = max(c.min_bank_rate_confidence, planar_conf)

        # ---------- 4-state constant-rate KF ----------
        F = np.array([
            [1.0, dt, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, dt],
            [0.0, 0.0, 0.0, 1.0],
        ])

        sigma_a = math.radians(c.terrain_ang_acc_sigma_degps2)
        qa = sigma_a * sigma_a
        q11 = dt**4 / 4.0 * qa
        q12 = dt**3 / 2.0 * qa
        q22 = dt**2 * qa
        Q = np.array([
            [q11, q12, 0.0, 0.0],
            [q12, q22, 0.0, 0.0],
            [0.0, 0.0, q11, q12],
            [0.0, 0.0, q12, q22],
        ])

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        sig_abs = math.radians(c.abs_angle_sigma_deg)
        sig_rate = math.radians(c.rate_sigma_degps)

        Rk = np.diag([
            sig_abs**2 / abs_conf_pitch,
            sig_rate**2 / rate_conf_pitch,
            sig_abs**2 / abs_conf_bank,
            sig_rate**2 / rate_conf_bank,
        ])

        z = np.array([
            theta_abs,
            theta_rate,
            phi_abs,
            phi_rate,
        ], dtype=float)

        # H = I
        S = self.P + Rk
        K = self.P @ np.linalg.inv(S)
        innovation = z - self.x

        clip = math.radians(c.innovation_clip_deg)
        innovation[0] = float(np.clip(innovation[0], -clip, clip))
        innovation[2] = float(np.clip(innovation[2], -clip, clip))

        self.x = self.x + K @ innovation

        I = np.eye(4)
        self.P = (I - K) @ self.P @ (I - K).T + K @ Rk @ K.T

        # The four residuals of a rectangular 4-point plane fit collapse to
        # the warp mode pattern [+W, -W, -W, +W]. We expose it for diagnosis.
        wheel_plane_residual = np.array([Wm, -Wm, -Wm, Wm])

        return {
            "road_pitch_rad": float(self.x[0]),
            "road_pitch_rate_radps": float(self.x[1]),
            "road_bank_rad": float(self.x[2]),
            "road_bank_rate_radps": float(self.x[3]),

            "theta_abs_meas_rad": float(theta_abs),
            "phi_abs_meas_rad": float(phi_abs),
            "theta_rate_meas_radps": float(theta_rate),
            "phi_rate_meas_radps": float(phi_rate),

            "susp_rel_pitch_rad": float(alpha_eff),
            "susp_rel_pitch_raw_rad": float(alpha_s),
            "pitch_geometry_gain": float(pitch_geometry_gain),
            "pitch_geometry_comp_rad": float(alpha_comp),
            "susp_rel_bank_rad": float(beta_eff),
            "susp_rel_bank_raw_rad": float(beta_s),
            "bank_geometry_gain": float(bank_geometry_gain),
            "bank_geometry_comp_rad": float(beta_comp),
            "warp_mode_m": float(Wm),
            "warp_rate_mps": float(warp_rate),
            "planar_confidence": float(planar_conf),
            "gravity_confidence": float(gravity_conf),

            "plane_residual_FL_m": float(wheel_plane_residual[0]),
            "plane_residual_FR_m": float(wheel_plane_residual[1]),
            "plane_residual_RL_m": float(wheel_plane_residual[2]),
            "plane_residual_RR_m": float(wheel_plane_residual[3]),
        }
