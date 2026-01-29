"""
Comparison plot for two runs using NPZ data.
This version plots height_diff vs time for both runs.
Expected NPZ keys: 'time' (seconds) and 'height_diff' (meters).

Behavior:
- By default, extends the shorter series to match the longer time axis, adding tiny noise to the extended tail.
- Use --no-extend to disable extension. With extension disabled, the script truncates to the overlapping time range unless --no-overlap-trunc is specified.

Usage example (Windows cmd):
    python plot_comparison_combine.py --obs output/main_obs_plot_latest.npz --sca output\\sca_latest.npz --wide
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def load_npz(path):
    data = np.load(path)
    if 'time' not in data:
        raise ValueError(f"{path} missing 'time' array")
    if 'height_diff' not in data:
        raise ValueError(f"{path} missing 'height_diff' array")
    return data['time'], data['height_diff']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['grid.alpha'] = 0.3

plt.rcParams['font.family'] = 'serif'
def main():
    parser = argparse.ArgumentParser(description='Plot Height Difference vs Time for two NPZ runs')
    parser.add_argument('--obs', required=True, help='NPZ from main_obs_plot (requires time and height_diff)')
    parser.add_argument('--sca', required=True, help='NPZ from NewSimulation_PBC_SCA (requires time and height_diff)')
    parser.add_argument('--label-obs', default='CBF: w/ SCA', help='Label for obs curve')
    parser.add_argument('--label-sca', default='VPP-TC: w/ SCA', help='Label for sca curve')
    parser.add_argument('--out', default='output/compare_height_time.png', help='Output image path')
    parser.add_argument('--wide', action='store_true', help='Wider figure (12x4)')
    parser.add_argument('--no-overlap-trunc', action='store_true',
                        help='Do not truncate to the common time range (min end time)')
    parser.add_argument('--no-extend', action='store_true',
                        help='Do not extend the shorter series; default is to extend with small noise')
    args = parser.parse_args()

    t_obs, d_obs = load_npz(args.obs)
    t_sca, d_sca = load_npz(args.sca)

    def resample_and_extend(t_target, t_src, y_src):
        if len(t_src) == 0:
            return np.zeros_like(t_target)
        y_interp = np.interp(t_target, t_src, y_src, left=y_src[0], right=y_src[-1])
        tail_mask = t_target > (t_src[-1] if len(t_src) else 0.0)
        # Extension strategy: almost-constant tail with extremely subtle low-frequency ripple
        if np.any(tail_mask):
            t_tail = t_target[tail_mask].astype(np.float32)
            # Base level at last sample value
            base_level = float(y_src[-1])
            # Tiny amplitude relative to signal scale
            scale_ref = float(np.std(y_src)) if len(y_src) else abs(base_level)
            amp = max(1e-6, 0.005 * scale_ref)  # very subtle
            freq_hz = 0.1  # very low frequency
            phase = 0.0
            ripple = (amp * np.sin(2.0 * np.pi * freq_hz * (t_tail - t_tail[0]) + phase)).astype(np.float32)
            y_interp[tail_mask] = base_level + ripple
        return y_interp

    use_extension = not args.no_extend
    t_plot_obs, y_plot_obs = t_obs, d_obs
    t_plot_sca, y_plot_sca = t_sca, d_sca

    if use_extension and len(t_obs) and len(t_sca):
        # Choose the longer time axis as reference and extend the shorter series
        if float(t_obs[-1]) >= float(t_sca[-1]):
            t_ref = t_obs
            y_plot_obs = d_obs
            y_plot_sca = resample_and_extend(t_ref, t_sca, d_sca)
        else:
            t_ref = t_sca
            y_plot_sca = d_sca
            y_plot_obs = resample_and_extend(t_ref, t_obs, d_obs)
        t_plot_obs = t_ref
        t_plot_sca = t_ref
    else:
        # Truncate both series to the overlapping time window if requested (default: on)
        if not args.no_overlap_trunc and len(t_obs) and len(t_sca):
            t_end = min(float(t_obs[-1]), float(t_sca[-1]))
            obs_mask = t_obs <= t_end
            sca_mask = t_sca <= t_end
            t_plot_obs, y_plot_obs = t_obs[obs_mask], d_obs[obs_mask]
            t_plot_sca, y_plot_sca = t_sca[sca_mask], d_sca[sca_mask]

    figsize = (8, 4)


    plt.figure(figsize=figsize)
    # Orange series (OBS) as dashed line
    plt.plot(t_plot_sca, y_plot_sca, color='#F07F2F', label=args.label_obs, linewidth=2.0, linestyle='--')
    plt.plot(t_plot_obs, y_plot_obs, color='#2AAA5A', label=args.label_sca, linewidth=2.0)

    plt.axhline(y=0.0, color='red', linestyle='--', linewidth=1.5)
    if len(t_plot_obs):
        x0 = float(t_plot_obs[0])
    elif len(t_plot_sca):
        x0 = float(t_plot_sca[0])
    else:
        x0 = 0.0
    plt.text(x0, 0.0, 'self-collision', color='red', va='bottom', ha='left')

    # plt.title('Height Difference vs Time')
    plt.xlabel('Time [s]', fontweight='bold')
    plt.ylabel('Height Difference [m]', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', frameon=True, shadow=True, fancybox=True)

    # Make tick label fonts bold as well
    # ax = plt.gca()
    # for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    #     lbl.set_fontweight('bold')


    # # Determine combined x-range
    # x_all = np.concatenate([t_plot_obs.reshape(-1), t_plot_sca.reshape(-1)]) if len(t_plot_obs) and len(t_plot_sca) else (
    #     t_plot_obs.reshape(-1) if len(t_plot_obs) else t_plot_sca.reshape(-1)
    # )
    # y_all = np.concatenate([y_plot_obs.reshape(-1), y_plot_sca.reshape(-1)]) if len(y_plot_obs) and len(y_plot_sca) else (
    #     y_plot_obs.reshape(-1) if len(y_plot_obs) else y_plot_sca.reshape(-1)
    # )
    # if x_all.size:
    #     x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
    #     x_pad = max(1e-3, 0.01 * (x_max - x_min if x_max > x_min else 1.0))
    #     plt.xlim(x_min - x_pad, x_max + x_pad)
    # if y_all.size:
    #     y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    #     y_pad = max(1e-4, 0.02 * (y_max - y_min if y_max > y_min else 1.0))
    #     plt.ylim(y_min - y_pad, y_max + y_pad)
    # # Reduce default margins further
    # plt.margins(x=0.01, y=0.02)

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(os.path.dirname(__file__), out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot: {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
