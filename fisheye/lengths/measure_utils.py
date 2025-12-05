import numpy as np
from matplotlib import pyplot as plt
import torch
from fisheye.utils import calculate_warped_points


def mapTokpt(heatmap, p=10, differentiable=False, round_to_integer=False):
    """
    heatmap: (N, K, H, W)
    returns x,y coordinates of the center of the heatmap
    """
    device = heatmap.device

    if differentiable:
        p_hmap = heatmap**p

        H = p_hmap.size(2)
        W = p_hmap.size(3)

        s_y = p_hmap.sum(3)  # (N, K, H)
        s_x = p_hmap.sum(2)  # (N, K, W)

        y = torch.arange(0, H, device=device)  # torch.linspace(-1.0, 1.0, H).cuda()
        x = torch.arange(0, W, device=device)  # torch.linspace(-1.0, 1.0, W).cuda()

        # u_y = (self.H_tensor * s_y).sum(2) / s_y.sum(2)  # (N, K)
        # u_x = (self.W_tensor * s_x).sum(2) / s_x.sum(2)
        u_y = (y * s_y).sum(2) / s_y.sum(2)  # (N, K)
        u_x = (x * s_x).sum(2) / s_x.sum(2)

    else:
        # find the brightest pixel in the heatmap
        B, C, H, W = heatmap.shape

        flat = heatmap.reshape(B, C, -1)  # B x C x (H*W)
        flat_idx = flat.argmax(dim=-1)  # B x C
        u_y = flat_idx // W  # B x C
        u_x = flat_idx % W  # B x C
        u_x = u_x.float()
        u_y = u_y.float()

    coords = torch.stack([u_x, u_y], dim=-1)
    if round_to_integer:
        coords = torch.round(coords)
    return coords  # , (var_x, var_y, cov)


def get_cone_edges(metadata, plot=False):
    # MAH 2025-11-26 20:36:52 TODO this doesnt perfectly line up with the actual cone edges, need to look into this
    corners_unwarped = [
        [0, 0],
        [0, metadata.unwarped_shape[0] - 1],
        [metadata.unwarped_shape[1] - 1, 0],
        [metadata.unwarped_shape[1] - 1, metadata.unwarped_shape[0] - 1],
    ]

    warped_points = calculate_warped_points(
        corners_unwarped,
        metadata,
        metadata.xdim,
        metadata.ydim,
    )

    ml, bl = np.polyfit(warped_points[:2, 0], warped_points[:2, 1], 1)
    mr, br = np.polyfit(warped_points[2:, 0], warped_points[2:, 1], 1)
    points_left = warped_points[:2]
    points_right = warped_points[2:]
    if plot:
        fig, ax = plt.subplots(figsize=(10, 10))
        dumby_frame = np.zeros((metadata.ydim, metadata.xdim))
        ax.imshow(dumby_frame, cmap="gray")
        ax.scatter(points_left[:, 0], points_left[:, 1], color="cyan")
        ax.scatter(points_right[:, 0], points_right[:, 1], color="magenta")

        plt.plot(warped_points[:2, 0], ml * warped_points[:2, 0] + bl, color="green")
        plt.plot(warped_points[2:, 0], mr * warped_points[2:, 0] + br, color="red")
        plt.show()
    return (
        points_left.tolist(),
        points_right.tolist(),
        [float(ml), float(bl)],
        [float(mr), float(br)],
    )


def windowed_mean(arr, window=5, min_count=1):
    """
    Centered moving average (boxcar) with NaN handling.

    Parameters
    ----------
    arr : array_like
        1D array (N,) or 2D array (N, D). For 2D, smoothing is done per column (dimension).
    window : int
        Total window length (should be a positive odd integer).
    min_count : int
        Minimum number of valid (non-NaN) values required to compute a mean at a position.
        If the available valid values in the window are fewer than this, result is NaN there.

    Returns
    -------
    smoothed : np.ndarray
        Same shape as arr. For 1D input, returns shape (N,).
        For 2D input, returns shape (N, D).
    """
    X = np.asarray(arr, dtype=float)
    squeeze_back = False
    if X.ndim == 1:
        X = X[:, None]  # treat as (N, 1)
        squeeze_back = True
    elif X.ndim != 2:
        raise ValueError("arr must be 1D or 2D (N,) or (N, D)")

    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")

    N, D = X.shape
    kernel = np.ones(window, dtype=float)

    # NaN-aware: track valid counts and sum only valid values
    valid = np.isfinite(X).astype(float)
    X_filled = np.where(np.isfinite(X), X, 0.0)

    sums = np.empty_like(X)
    counts = np.empty_like(X)

    for d in range(D):
        sums[:, d] = np.convolve(X_filled[:, d], kernel, mode="same")
        counts[:, d] = np.convolve(valid[:, d], kernel, mode="same")

    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts

    # Enforce min_count (positions with too few valid samples -> NaN)
    means[counts < float(min_count)] = np.nan

    if squeeze_back:
        means = means[:, 0]
    return means


def robust_mean(values, frame_nums, window_size=5, z_thresh=2.5):
    """
    Robust moving average using median + MAD, accounting for irregular or missing frame indices.

    values: list/array of numbers
    frame_nums: list of frame/time indices (same length as values)
    window_size: time radius in frame units (not sample count)
    z_thresh: how many MADs away counts as an outlier

    Returns:
        A list of smoothed values, same order as input.
    """
    import math
    from bisect import bisect_left, bisect_right

    assert len(values) == len(frame_nums), "values and frame_nums must be same length"
    n = len(values)
    if n == 0:
        return []

    # Helper: median
    def median(lst):
        s = sorted(lst)
        m = len(s)
        return s[m // 2] if (m % 2 == 1) else 0.5 * (s[m // 2 - 1] + s[m // 2])

    # Sort by frame number (for bisect)
    order = sorted(range(n), key=lambda i: frame_nums[i])
    frames_sorted = [frame_nums[i] for i in order]
    vals_sorted = [values[i] for i in order]

    result_sorted = [math.nan] * n

    for k in range(n):
        t = frames_sorted[k]
        # Window covers all samples within ±window_size in frame units
        left = bisect_left(frames_sorted, t - window_size)
        right = bisect_right(frames_sorted, t + window_size)
        window = vals_sorted[left:right]

        if not window:
            result_sorted[k] = math.nan
            continue

        # Median
        med = median(window)

        # MAD (median absolute deviation)
        abs_dev = [abs(x - med) for x in window]
        mad = median(abs_dev)

        # Inliers based on z-score
        if mad == 0:
            inliers = window
        else:
            inliers = [x for x in window if abs(0.6745 * (x - med) / mad) <= z_thresh]

        # Average inliers (fallback to median if none)
        if inliers:
            result_sorted[k] = sum(inliers) / len(inliers)
        else:
            result_sorted[k] = med

    # Restore original order
    inv_order = [0] * n
    for pos, orig_i in enumerate(order):
        inv_order[orig_i] = pos

    result = [result_sorted[inv_order[i]] for i in range(n)]
    return result


def robust_mean_vector(vx, vy, frame_nums, window_size=5, z_thresh=2.5):
    """
    Robust moving average for 2D vectors using median + MAD on radial distance.
    Uses `frame_nums` (time indices) to form a variable-size window consisting of
    all points with frame indices within ±window_size of the current point.

    Returns four lists (aligned to the ORIGINAL input order):
      - mxs, mys: per-point median vector components within the time window
      - sxs, sys: per-point robust-mean (inlier-avg) vector components

    Notes:
      * `window_size` is a radius in FRAME UNITS, not a sample count.
      * Handles irregular sampling and missing frames.
    """
    import math
    from bisect import bisect_left, bisect_right

    assert (
        len(vx) == len(vy) == len(frame_nums)
    ), "vx, vy, frame_nums must be same length"
    n = len(vx)
    if n == 0:
        return [], [], [], []

    # Helper: median for a non-empty list
    def median(lst):
        s = sorted(lst)
        m = len(s)
        return s[m // 2] if (m % 2 == 1) else 0.5 * (s[m // 2 - 1] + s[m // 2])

    # Sort by frame index (in case user passes unsorted inputs)
    order = sorted(range(n), key=lambda i: frame_nums[i])
    frames_sorted = [frame_nums[i] for i in order]
    vx_sorted = [vx[i] for i in order]
    vy_sorted = [vy[i] for i in order]

    # Prepare outputs in sorted order; we'll unsort at the end
    mxs_s, mys_s, sxs_s, sys_s = [None] * n, [None] * n, [None] * n, [None] * n

    for k in range(n):
        t = frames_sorted[k]
        # window is all indices j with frame in [t - window_size, t + window_size]
        left = bisect_left(frames_sorted, t - window_size)
        right = bisect_right(frames_sorted, t + window_size)  # exclusive
        wx = vx_sorted[left:right]
        wy = vy_sorted[left:right]

        # median vector
        mx = median(wx)
        my = median(wy)

        # distances from median vector
        dists = [math.hypot(x - mx, y - my) for x, y in zip(wx, wy)]

        # MAD of distances
        md = median(dists)
        abs_dev = [abs(d - md) for d in dists]
        mad = median(abs_dev)

        # choose inliers by radial z-score
        if mad == 0:
            inliers = list(zip(wx, wy))
        else:
            inliers = []
            for (x, y), d in zip(zip(wx, wy), dists):
                z = 0.6745 * (d - md) / mad
                if abs(z) <= z_thresh:
                    inliers.append((x, y))

        if inliers:
            sx = sum(p[0] for p in inliers) / len(inliers)
            sy = sum(p[1] for p in inliers) / len(inliers)
        else:
            # fall back to median if everything was rejected
            sx, sy = mx, my

        mxs_s[k], mys_s[k], sxs_s[k], sys_s[k] = mx, my, sx, sy

    # Unsort back to original order
    inv_order = [0] * n
    for pos, orig_i in enumerate(order):
        inv_order[orig_i] = pos

    mxs = [mxs_s[inv_order[i]] for i in range(n)]
    mys = [mys_s[inv_order[i]] for i in range(n)]
    sxs = [sxs_s[inv_order[i]] for i in range(n)]
    sys = [sys_s[inv_order[i]] for i in range(n)]

    return mxs, mys, sxs, sys


def get_velocity_dev(
    pred_kpts_global_0, pred_kpts_global_1, frame_nums, window_size=5, plot=False
):
    velocity_deltas = np.empty([2, len(pred_kpts_global_0)])
    velocities_xsht = []
    velocities_ysht = []
    sxsht = []
    sysht = []
    for ht_i, pred_kpts_global in enumerate([pred_kpts_global_0, pred_kpts_global_1]):
        velocities_x = []
        velocities_y = []
        for i in range(len(pred_kpts_global)):
            if i == len(pred_kpts_global) - 1:
                velocities_x.append(velocities_x[-1])
                velocities_y.append(velocities_y[-1])
                continue

            vx = pred_kpts_global[i + 1][0] - pred_kpts_global[i][0]
            vy = pred_kpts_global[i + 1][1] - pred_kpts_global[i][1]
            v = np.linalg.norm(vx) + np.linalg.norm(vy)
            velocities_x.append(vx)
            velocities_y.append(vy)

        mxs, mys, sxs, sys = robust_mean_vector(
            velocities_x, velocities_y, frame_nums, window_size=window_size
        )

        velocities_deviation = np.hypot(
            np.array(velocities_x) - np.array(sxs),
            np.array(velocities_y) - np.array(sys),
        )
        velocity_deltas[ht_i] = velocities_deviation

        if plot:
            fig, ax = plt.subplots(2, 3, figsize=(12, 6))
            ax = ax.flatten()
            ax[0].plot(velocities_x, label="velocities_x")
            ax[0].plot(mxs, label="mxs")
            ax[0].plot(sxs, label="sxs")
            ax[1].plot(velocities_y, label="velocities_y")
            ax[1].plot(mys, label="mys")
            ax[1].plot(sys, label="sys")
            ax[2].plot(velocities_deviation, label="velocities_deviation")

            ax[3].plot(np.array(velocities_x) - np.array(sxs), label="velocities_x-sxs")
            ax[4].plot(np.array(velocities_y) - np.array(sys), label="velocities_y-sys")
            ax[5].plot(velocities_deviation, label="velocities_deviation")

            ax[0].legend()
            ax[1].legend()
            ax[2].legend()
            ax[3].legend()
            ax[4].legend()
            ax[5].legend()
            plt.show()

            velocities_xsht.append(velocities_x)
            velocities_ysht.append(velocities_y)
            sxsht.append(sxs)
            sysht.append(sys)
    if plot:

        fig, ax = plt.subplots(2, 3, figsize=(12, 12))
        for ii, (velocities_xs, velocities_ys, sxs, syss) in enumerate(
            zip(velocities_xsht, velocities_ysht, sxsht, sysht)
        ):
            for i, (sx, sy) in enumerate(zip(sxs, syss)):
                col = plt.cm.viridis(i / len(velocities_xs))
                ax[ii, 0].plot([0, sx], [0, sy], label="velocitiesxy", color=col)
            ax[ii, 0].set_ylim(ax[ii, 0].get_ylim()[1], ax[ii, 0].get_ylim()[0])
            ax[ii, 0].set_title(f"smoothed vel {ii}")

            for i, (vx, vy, sx, sy) in enumerate(
                zip(velocities_xs, velocities_ys, sxs, syss)
            ):
                col = plt.cm.viridis(i / len(velocities_x))
                ax[ii, 1].plot([sx, vx], [sy, vy], label="velocitiesxy", color=col)
            ax[ii, 1].set_ylim(ax[ii, 1].get_ylim()[1], ax[ii, 1].get_ylim()[0])
            ax[ii, 1].set_title(f"velocitiesxy {ii}")

            cum_x = 0
            cum_y = 0
            cum_sx = 0
            cum_sy = 0
            for i in range(len(velocities_xs) - 1):
                col = plt.cm.viridis(i / len(velocities_xs))
                ax[ii, 2].plot(
                    [cum_x, cum_x + velocities_xs[i]],
                    [cum_y, cum_y + velocities_ys[i]],
                    color=col,
                )
                ax[ii, 2].plot(
                    [cum_sx, cum_sx + sxs[i]],
                    [cum_sy, cum_sy + syss[i]],
                    color=col,
                    linestyle="--",
                )
                cum_x += velocities_xs[i]
                cum_y += velocities_ys[i]
                cum_sx += sxs[i]
                cum_sy += syss[i]
            ax[ii, 2].set_ylim(ax[ii, 2].get_ylim()[1], ax[ii, 2].get_ylim()[0])
            ax[ii, 2].set_title(f"cumulative vel {ii}")

        plt.show()
    velocity_deltas = np.max(velocity_deltas, axis=0)

    return velocity_deltas


def get_deviation_from_average(vectors, frame_nums, window_size=5, robust=False):
    vector_averages = windowed_mean(np.array(vectors), window=window_size)
    if robust:
        vector_averages = robust_mean(vectors, frame_nums, window_size=5, z_thresh=2.5)
    change_in = [vectors[i] - vector_averages[i] for i in range(len(vectors))]
    return change_in, vector_averages


def get_change_in_length(
    pred_kpts_global_0, pred_kpts_global_1, frame_nums, window_size=5, robust=False
):
    length_vectors = [
        np.linalg.norm(p1 - p0)
        for p0, p1 in zip(pred_kpts_global_0, pred_kpts_global_1)
    ]

    change_in_length, length_vector_averages = get_deviation_from_average(
        length_vectors, frame_nums, window_size=window_size, robust=robust
    )

    return change_in_length, length_vector_averages


def calc_len(kpts):
    is_torch = True if isinstance(kpts, torch.Tensor) else False
    if is_torch:
        if kpts.ndim == 3:
            return torch.linalg.norm(kpts[:, 1] - kpts[:, 0], dim=-1)
        else:
            return torch.linalg.norm(kpts[1] - kpts[0], dim=-1)
    else:
        if kpts.ndim == 3:
            return np.linalg.norm(kpts[:, 1] - kpts[:, 0], axis=-1)
        else:
            return np.linalg.norm(kpts[1] - kpts[0])


def point_line_distance(point, m, b, side="above"):
    """
    Computes signed perpendicular distance from a point to a line y = m*x + b.

    point : (x0, y0)
    m, b  : slope and intercept of line
    side  : 'above' or 'below' – determines which side is considered positive.
    # MAH 2025-11-05 17:13:31 rememvber that images y axis is inverted
    returns: signed distance (float)
    """
    x0, y0 = point
    # signed distance (no absolute value)
    signed_dist = (m * x0 - y0 + b) / np.sqrt(m**2 + 1)

    # convention: if side='above', then points above y=m*x+b are positive
    # adjust sign based on chosen convention
    if side == "above":
        return -signed_dist
    elif side == "below":
        return signed_dist
    else:
        raise ValueError("side must be 'above' or 'below'")


def foot_to_line(point, m, b):
    """Orthogonal projection of (x0,y0) onto y = m x + b."""
    x0, y0 = point
    denom = m * m + 1.0
    xp = (x0 + m * (y0 - b)) / denom
    yp = m * xp + b
    return np.array([xp, yp], dtype=float)


def get_min_edge_distances_pxl(pred_kpts_global, ml, bl, mr, br):
    """
    Args:
      pred_kpts_global: (N,2) array-like or torch.Tensor of (x,y) points
      ml, bl: slope/intercept of left line  (y = ml*x + bl)
      mr, br: slope/intercept of right line (y = mr*x + br)
    Returns:
      min_edge_distance: float, smallest distance among all points to either line
      per_point: list of dicts, one per point:
         {
           'point': (x, y),
           'closest_line': 'left' or 'right',
           'distance': float,
           'foot': (xp, yp)   # projection on the closest line
         }
    """
    # Handle torch input without importing torch here
    if hasattr(pred_kpts_global, "detach") and hasattr(pred_kpts_global, "cpu"):
        pred_kpts_global = pred_kpts_global.detach().cpu().numpy()

    pts = np.asarray(pred_kpts_global, dtype=float).reshape(-1, 2)

    per_point = []
    min_edge_distance = None

    for pt in pts:
        d_left = point_line_distance(pt, ml, bl, side="below")
        d_right = point_line_distance(pt, mr, br, side="below")

        if d_left <= d_right:
            closest = "left"
            d = d_left
            foot = foot_to_line(pt, ml, bl)
        else:
            closest = "right"
            d = d_right
            foot = foot_to_line(pt, mr, br)

        per_point.append(
            {
                "point": (float(pt[0]), float(pt[1])),
                "closest_line": closest,
                "distance": float(d),
                "foot": (float(foot[0]), float(foot[1])),
            }
        )

        if (min_edge_distance is None) or (d < min_edge_distance):
            min_edge_distance = float(d)

    return min_edge_distance, per_point


def _to_luma(img):
    """
    Convert HxW, HxWx3, or HxWx4 image to a 2D float array (luma/brightness).
    Uses Rec.709 luma weights on sRGB data (common practical approximation).
    """
    img = np.asarray(img)
    if img.ndim == 2:  # already grayscale
        return img.astype(float)
    if img.ndim == 3 and img.shape[2] >= 3:
        r = img[..., 0].astype(float)
        g = img[..., 1].astype(float)
        b = img[..., 2].astype(float)
        # Rec.709 luma on sRGB values (no explicit linearization).
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    raise ValueError("Unsupported image shape; expected HxW or HxWxC with C>=3.")


def _bresenham_line(p0, p1):
    """
    Integer pixel coordinates along a line from p0 to p1 inclusive (x, y).
    p0, p1: (x, y)
    Returns: (N, 2) array of integer coords.
    """
    x0, y0 = map(int, map(round, p0))
    x1, y1 = map(int, map(round, p1))
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    pts = []
    x, y = x0, y0
    while True:
        pts.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return np.array(pts, dtype=int)


def _bilinear_sample(gray, xs, ys):
    """
    Bilinear sampling at fractional coords xs, ys on 2D array 'gray'.
    Outside-edge samples are clamped to valid borders.
    """
    h, w = gray.shape
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    # Clamp to [0, w-1] / [0, h-1]
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)

    x0 = np.floor(xs).astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.floor(ys).astype(int)
    y1 = np.clip(y0 + 1, 0, h - 1)

    dx = xs - x0
    dy = ys - y0

    Ia = gray[y0, x0]
    Ib = gray[y0, x1]
    Ic = gray[y1, x0]
    Id = gray[y1, x1]

    # Bilinear interpolation
    top = Ia * (1 - dx) + Ib * dx
    bot = Ic * (1 - dx) + Id * dx
    return top * (1 - dy) + bot * dy


def average_brightness_on_line(image, p0, p1, method="bilinear", samples=None):
    """
    Compute average brightness along the line from p0 to p1.

    Parameters
    ----------
    image : np.ndarray
        Grayscale (H,W) or RGB/RGBA (H,W,3/4). Any dtype (uint8/float etc).
    p0, p1 : tuple(float, float)
        Start and end coordinates as (x, y). Can be non-integer.
    method : {"bilinear", "bresenham"}
        - "bilinear": subpixel sampling using bilinear interpolation.
        - "bresenham": integer-pixel sampling along the line.
    samples : int or None
        For "bilinear": number of samples along the line. If None, uses
        ceil(euclidean_length) for ~1 sample per pixel of length.

    Returns
    -------
    float
        Mean brightness along the line.
    """
    gray = _to_luma(image)

    if method == "bresenham":
        pts = _bresenham_line(p0, p1)  # integer (x, y)
        # Filter points inside image bounds
        h, w = gray.shape
        mask = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
        pts = pts[mask]
        if pts.size == 0:
            return np.nan
        vals = gray[pts[:, 1], pts[:, 0]]
        return float(np.mean(vals))

    elif method == "bilinear":
        # Parametric line sampling
        x0, y0 = map(float, p0)
        x1, y1 = map(float, p1)
        length = np.hypot(x1 - x0, y1 - y0)
        n = int(np.ceil(length)) if samples is None else int(samples)
        n = max(n, 2)  # at least two points (endpoints)
        t = np.linspace(0.0, 1.0, n)
        xs = x0 + (x1 - x0) * t
        ys = y0 + (y1 - y0) * t
        vals = _bilinear_sample(gray, xs, ys)
        return float(np.mean(vals))

    else:
        raise ValueError("method must be 'bilinear' or 'bresenham'")


def vis_3_channel_img(img, normalize=False):
    assert img.shape[0] == 3, f"{img.shape=}"
    if normalize:
        img -= torch.min(img)
        img /= torch.max(img)
    img[0] = torch.max(img[0], img[1])
    img[2] = torch.max(img[1], img[2])

    return img.permute(1, 2, 0)
