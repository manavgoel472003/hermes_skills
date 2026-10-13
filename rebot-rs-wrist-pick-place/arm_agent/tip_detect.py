"""Robust orange-fingertip detector for the tag-free DLT camera calibration.

The tip point we want is the physical grasp point (the TCP the arm reports in
end_effector.position_m). With the gripper CLOSED the two orange fins converge to
a point at the BOTTOM of the blob -- that convergence is our tip proxy. So we:
  * threshold a TIGHT orange hue (excludes the red finger plates, which are hue
    ~0-6 / 175+, and any warm background),
  * keep the largest 1-2 orange regions (the two fins / merged blob),
  * take the tip pixel = centroid of the LOWEST slice of those orange pixels.
This deliberately targets the distal tip, not the gripper body (the old bug).
"""
from __future__ import annotations

import cv2
import numpy as np

# Tight orange: hue 8-24 skips red (<=6) and yellow (>=28); high sat+val = the
# saturated painted fins, not dull background.
_LO = np.array([8, 120, 110], np.uint8)
_HI = np.array([24, 255, 255], np.uint8)


def orange_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, _LO, _HI)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def detect_tip(bgr: np.ndarray, min_area: float = 60.0, bottom_frac: float = 0.18):
    """Return (u, v, info) for the fingertip, or (None, None, info).

    info carries the mask pixel count and blob area for debugging/QC.
    """
    m = orange_mask(bgr)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > min_area]
    info = {"n_contours": len(cnts), "mask_px": int(m.sum() // 255)}
    if not cnts:
        return None, None, info
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]
    info["area"] = float(sum(cv2.contourArea(c) for c in cnts))
    # Fill the chosen blobs, then take the centroid of their LOWEST slice.
    blob = np.zeros(m.shape, np.uint8)
    cv2.drawContours(blob, cnts, -1, 255, -1)
    ys, xs = np.where(blob > 0)
    y_lo = ys.max() - max(2.0, bottom_frac * (ys.max() - ys.min()))
    sel = ys >= y_lo
    u = float(xs[sel].mean())
    v = float(ys[sel].mean())
    info["y_span"] = [int(ys.min()), int(ys.max())]
    return u, v, info


def detect_grasp_center(bgr: np.ndarray, min_area: float = 60.0):
    """Return (u, v, info) for the GRASP CENTER -- the point that must land on the
    object. With the gripper OPEN the two orange fins are separate: the grasp
    center is the midpoint of their centroids. With one blob (closed) it's that
    blob's centroid. This is what we servo onto the object's pixel.
    """
    m = orange_mask(bgr)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > min_area]
    info = {"n_contours": len(cnts), "mask_px": int(m.sum() // 255)}
    if not cnts:
        return None, None, info
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]
    info["area"] = float(sum(cv2.contourArea(c) for c in cnts))
    cents = []
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"] > 0:
            cents.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
    if not cents:
        return None, None, info
    info["n_fins"] = len(cents)
    u = float(np.mean([c[0] for c in cents]))
    v = float(np.mean([c[1] for c in cents]))
    return u, v, info


def annotate(bgr: np.ndarray, u, v, info) -> np.ndarray:
    out = bgr.copy()
    m = orange_mask(bgr)
    out[m > 0] = (0, 165, 255)  # paint detected orange
    if u is not None:
        cv2.drawMarker(out, (int(u), int(v)), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
        cv2.circle(out, (int(u), int(v)), 8, (0, 255, 0), 2)
        cv2.putText(out, f"tip ({u:.0f},{v:.0f}) area={info.get('area',0):.0f}",
                    (int(u) - 120, int(v) + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    img = cv2.imread(path)
    u, v, info = detect_tip(img)
    print("tip:", u, v, "info:", info)
    out = annotate(img, u, v, info)
    outp = path.rsplit(".", 1)[0] + "_tip.jpg"
    cv2.imwrite(outp, out)
    print("wrote", outp)
