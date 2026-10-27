
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np

from wa_rsbe.observer import ObserverConfig, WarpAwareSuspensionObserver


def cmap(names):
    return {str(v): i for i, v in enumerate(names.tolist())}


def required_row(row, xmap):
    return {
        name: float(row[xmap[name]])
        for name in WarpAwareSuspensionObserver.REQUIRED
    }


def check_columns(xmap):
    missing = [x for x in WarpAwareSuspensionObserver.REQUIRED if x not in xmap]
    if missing:
        raise KeyError(f"Missing required X26 channels: {missing}")


def rmse_deg(a, b, m):
    if not np.any(m):
        return float("nan")
    e = a[m] - b[m]
    return float(np.degrees(np.sqrt(np.mean(e * e))))


def mae_deg(a, b, m):
    if not np.any(m):
        return float("nan")
    return float(np.degrees(np.mean(np.abs(a[m] - b[m]))))


def process_case(X, x_columns, warmup, valid, cfg, mode,
                 Y=None, y_columns=None, event_code=None):
    xmap = cmap(x_columns)
    check_columns(xmap)

    obs = WarpAwareSuspensionObserver(cfg, mode=mode)

    nw = max(5, min(int(warmup), len(X) - 1))
    warm_rows = [required_row(X[k], xmap) for k in range(nw)]
    obs.initialize_from_warmup(warm_rows)

    rows = []
    for k in range(len(X)):
        rows.append(obs.step(required_row(X[k], xmap)))

    names = list(rows[0].keys())
    A = np.asarray([[r[n] for n in names] for r in rows], dtype=np.float64)

    metrics = {}
    if Y is not None and y_columns is not None:
        ymap = cmap(y_columns)
        m = np.asarray(valid, dtype=bool).copy()
        m[:nw] = False

        if "road_pitch_rad" in ymap:
            ep = A[:, names.index("road_pitch_rad")]
            tp = Y[:, ymap["road_pitch_rad"]]
            metrics["pitch_rmse_deg"] = rmse_deg(ep, tp, m)
            metrics["pitch_mae_deg"] = mae_deg(ep, tp, m)

        if "road_bank_rad" in ymap:
            eb = A[:, names.index("road_bank_rad")]
            tb = Y[:, ymap["road_bank_rad"]]
            metrics["bank_rmse_deg"] = rmse_deg(eb, tb, m)
            metrics["bank_mae_deg"] = mae_deg(eb, tb, m)

        # OFFLINE evaluation only. event_code NEVER enters the observer.
        if event_code is not None:
            for ev in np.unique(event_code[m]):
                em = m & (event_code == ev)
                if "road_pitch_rad" in ymap:
                    metrics[f"event_{int(ev)}_pitch_rmse_deg"] = rmse_deg(ep, tp, em)
                if "road_bank_rad" in ymap:
                    metrics[f"event_{int(ev)}_bank_rmse_deg"] = rmse_deg(eb, tb, em)

    return A, names, metrics


def iter_npz(zip_path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if not name.endswith(".npz"):
                continue
            raw = zf.read(name)
            with np.load(io.BytesIO(raw), allow_pickle=True) as d:
                yield name, {
                    k: d[k].copy() if isinstance(d[k], np.ndarray) else d[k]
                    for k in d.files
                }


def plot_case(case_id, time_s, A, names, Y, y_columns, event_code, save_to):
    import matplotlib.pyplot as plt

    ymap = cmap(y_columns)
    p_est = np.degrees(A[:, names.index("road_pitch_rad")])
    b_est = np.degrees(A[:, names.index("road_bank_rad")])
    warp_mm = 1000.0 * A[:, names.index("warp_mode_m")]
    cp = A[:, names.index("planar_confidence")]
    cg = A[:, names.index("gravity_confidence")]

    fig = plt.figure(figsize=(11, 10))

    ax1 = fig.add_subplot(4, 1, 1)
    ax1.plot(time_s, np.degrees(Y[:, ymap["road_pitch_rad"]]), label="truth")
    ax1.plot(time_s, p_est, label="estimate")
    ax1.set_ylabel("pitch (deg)")
    ax1.grid(True)
    ax1.legend()

    ax2 = fig.add_subplot(4, 1, 2)
    ax2.plot(time_s, np.degrees(Y[:, ymap["road_bank_rad"]]), label="truth")
    ax2.plot(time_s, b_est, label="estimate")
    ax2.set_ylabel("bank (deg)")
    ax2.grid(True)
    ax2.legend()

    ax3 = fig.add_subplot(4, 1, 3)
    ax3.plot(time_s, warp_mm)
    ax3.set_ylabel("warp (mm)")
    ax3.grid(True)

    ax4 = fig.add_subplot(4, 1, 4)
    ax4.plot(time_s, cp, label="planar confidence")
    ax4.plot(time_s, cg, label="gravity confidence")
    if event_code is not None:
        emax = max(1, int(np.max(event_code)))
        ax4.plot(time_s, event_code / emax, alpha=0.45, label="event/max")
    ax4.set_ylim(-0.05, 1.05)
    ax4.set_ylabel("confidence")
    ax4.set_xlabel("time (s)")
    ax4.grid(True)
    ax4.legend()

    fig.suptitle(case_id)
    fig.tight_layout()
    fig.savefig(save_to, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", nargs="+", required=True)
    ap.add_argument("--out", default="WA_NO_VX_results")
    ap.add_argument("--mode", default="full",
                    choices=["full", "suspension_no_warp", "imu_only"])
    ap.add_argument("--wheelbase-m", type=float, default=3.0)
    ap.add_argument("--track-m", type=float, default=1.71)
    ap.add_argument("--motion-ratio", type=float, default=1.0,
                    help="Physical suspension sensor/wheel motion ratio. Do NOT fit with Y truth.")
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--max-plots", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "cases").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    cfg = ObserverConfig(
        wheelbase_m=args.wheelbase_m,
        track_m=args.track_m,
        suspension_motion_ratio=args.motion_ratio,
    )

    summary = []
    n = 0
    nplots = 0

    for zp in args.zip:
        zp = Path(zp)
        for part_name, d in iter_npz(zp):
            X = d["X26"]
            xcols = d["X26_columns"]
            Y = d.get("Y")
            ycols = d.get("Y_columns")
            valid = d.get("valid_mask", np.ones(X.shape[:2], bool))
            events = d.get("event_code")
            warmup = int(np.asarray(d.get("warmup_steps", 100)).item())
            ids = d.get("source_case_ids", np.array([f"case_{i}" for i in range(X.shape[0])]))
            time_s = d.get("time_s", np.arange(X.shape[1]) * cfg.dt)

            for i in range(X.shape[0]):
                case_id = str(ids[i])
                event_i = events[i] if events is not None else None

                A, names, metrics = process_case(
                    X[i], xcols, warmup, valid[i], cfg, args.mode,
                    Y=Y[i] if Y is not None else None,
                    y_columns=ycols,
                    event_code=event_i,
                )

                np.savez_compressed(
                    out / "cases" / f"{case_id}_{args.mode}.npz",
                    estimates=A,
                    estimate_columns=np.asarray(names),
                    time_s=time_s,
                    case_id=case_id,
                    mode=args.mode,
                    online_contract=(
                        "IMU6_PLUS_SUSP4_ONLY__NO_VX_NO_WHEEL_SPEED_NO_Y_"
                        "NO_PRIVILEGED_NO_FUTURE"
                    ),
                )

                summary.append({
                    "case_id": case_id,
                    "source_zip": zp.name,
                    "part": part_name,
                    "mode": args.mode,
                    **metrics,
                })

                if nplots < args.max_plots and Y is not None and ycols is not None:
                    plot_case(
                        case_id, time_s, A, names, Y[i], ycols, event_i,
                        out / "plots" / f"{case_id}_{args.mode}.png",
                    )
                    nplots += 1

                n += 1
                print(f"[{n}] {zp.name} {case_id}: {metrics}")

                if args.max_cases and n >= args.max_cases:
                    break

            if args.max_cases and n >= args.max_cases:
                break
        if args.max_cases and n >= args.max_cases:
            break

    fields = []
    for row in summary:
        for k in row:
            if k not in fields:
                fields.append(k)

    with open(out / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)

    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump({**vars(cfg), "mode": args.mode}, f, ensure_ascii=False, indent=2)

    print(f"\nDone: {n} cases")
    print(f"Summary: {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
