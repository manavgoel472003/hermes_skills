"""HTTP client for the rebot_arm_service backend.

Wraps the endpoints the planner needs into a small typed surface. Everything the
brain does to the arm goes through here, so the service stays the single owner of
the hardware and the single place where safety clamps live.
"""

from __future__ import annotations

import os
from typing import Any

import httpx


class ArmServiceClient:
    def __init__(self, base_url: str | None = None, timeout: float = 120.0) -> None:
        self.base_url = (base_url or os.getenv("REBOT_ARM_SERVICE_URL", "http://127.0.0.1:8000")).rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ArmServiceClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- low level ----------------------------------------------------------
    def _get(self, path: str, **params: Any) -> dict[str, Any]:
        r = self._client.get(path, params={k: v for k, v in params.items() if v is not None})
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, json: Any = None, **params: Any) -> dict[str, Any]:
        r = self._client.post(path, json=json, params={k: v for k, v in params.items() if v is not None})
        r.raise_for_status()
        return r.json()

    # -- health / state -----------------------------------------------------
    def healthz(self) -> dict[str, Any]:
        return self._get("/healthz")

    def vision_health(self) -> dict[str, Any]:
        return self._get("/vision/health")

    def connect(self) -> dict[str, Any]:
        return self._post("/connect")

    def state(self) -> dict[str, Any]:
        return self._get("/state")

    # -- perception ---------------------------------------------------------
    def scene(self, camera: str = "top", max_objects: int = 20) -> dict[str, Any]:
        return self._get("/scene", camera=camera, max_objects=max_objects)

    def locate(self, description: str, camera: str = "top") -> dict[str, Any]:
        return self._post("/vision/locate", json={"description": description, "camera": camera})

    def calibration(self) -> dict[str, Any]:
        return self._get("/scene/calibration")

    # -- skills -------------------------------------------------------------
    def skill_list(self) -> dict[str, Any]:
        return self._get("/skill/list")

    def run_skill(self, name: str, params: dict[str, Any] | None = None,
                  dry_run: bool = False) -> dict[str, Any]:
        return self._post(f"/skill/{name}", json=params or {}, dry_run=str(dry_run).lower())

    # -- low-level motion (used by the run_code escape-hatch, M5) ------------
    def move_pose(self, x: float, y: float, z: float, roll: float = 0.0,
                  pitch: float = 0.0, yaw: float = 0.0, mode: str = "traj",
                  duration: float = 2.5) -> dict[str, Any]:
        return self._post("/move/pose", json={
            "x": x, "y": y, "z": z, "roll": roll, "pitch": pitch, "yaw": yaw,
            "mode": mode, "duration": duration,
        })

    def gripper(self, action: str) -> dict[str, Any]:
        return self._post("/gripper", json={"action": action})

    def run_code(self, code: str, dry_run: bool = False) -> dict[str, Any]:
        return self._post("/code/run", json={"code": code, "dry_run": dry_run})

    def home(self) -> dict[str, Any]:
        return self._post("/home")
