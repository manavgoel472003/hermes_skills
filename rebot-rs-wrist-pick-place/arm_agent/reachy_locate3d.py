"""Locate a named object in the B601 base frame from the Reachy camera:
detection (LocateAnything) -> pixel; homography (arm-calibrated) -> table (x,y);
DepthAnything -> height signal for elevated objects. Run after reachy calibrate.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/manav/rebot_lerobot")
sys.path.insert(0, "/home/manav/rebot_lerobot/rebot_arm_service/src")
from rebot_arm_service.scene import TableCalibration, LocateAnythingLocator
from rebot_arm_service.scene.depth_anything import DepthEstimator
from arm_agent.reachy_client import ReachyBridgeClient

CALIB = Path("/home/manav/rebot_lerobot/rebot_arm_service/calib")
_la = None; _de = None

def _models():
    global _la, _de
    if _la is None: _la = LocateAnythingLocator()
    if _de is None: _de = DepthEstimator()
    return _la, _de

def locate3d(obj: str, reachy: ReachyBridgeClient | None = None):
    meta = json.load(open(CALIB / "reachy_calibration_meta.json"))
    gv = meta["grasp_view_head_pose"]
    cal = TableCalibration.load(str(CALIB / "reachy_cam.json"))
    reachy = reachy or ReachyBridgeClient()
    reachy.head(pose=gv, duration=1.2)
    frame = reachy.frame_bgr(warmup=6)
    la, de = _models()
    r = la.locate(frame, obj)
    if not r.get("found"):
        return {"ok": True, "found": False, "object": obj}
    b = r["box"]; u = (b[0]+b[2])/2.0; v = b[3]          # bottom-centre = table contact
    x, y = cal.pixel_to_table(u, v)
    dm = de.depth(frame)
    dobj = DepthEstimator.box_depth(dm, b)
    # local table depth = median of a ring just outside the box
    import numpy as np
    h, w = dm.shape; pad = 20
    x1,y1,x2,y2 = [int(t) for t in b]
    ring = np.concatenate([dm[max(0,y1-pad):y1, x1:x2].ravel(), dm[y2:min(h,y2+pad), x1:x2].ravel()]) if y2<h else np.array([dobj])
    tbl = float(np.median(ring)) if ring.size else dobj
    return {"ok": True, "found": True, "object": obj,
            "base_xy_m": [round(float(x),3), round(float(y),3)],
            "pixel": [round(u,0), round(v,0)],
            "depth_obj": round(float(dobj),2), "depth_table": round(tbl,2),
            "elevated": bool(dobj - tbl > 0.3)}

if __name__ == "__main__":
    import sys
    print(locate3d(sys.argv[1] if len(sys.argv)>1 else "red cube"))
