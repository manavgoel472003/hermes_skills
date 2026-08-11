"""Bounded joint-space RRT-Connect used only as a collision-route proposer.

The caller owns robot semantics and remains the safety authority.  This module
knows only joint bounds and two read-only predicates: state validity and edge
validity.  Every returned path must still be certified by the hardware backend
immediately before execution.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np


StatePredicate = Callable[[np.ndarray], bool]
EdgePredicate = Callable[[np.ndarray, np.ndarray], bool]


@dataclass(frozen=True)
class PlannerSettings:
    timeout_s: float = 4.0
    max_nodes: int = 4000
    step_rad: float = 0.12
    edge_step_rad: float = 0.03
    goal_bias: float = 0.15
    shortcut_attempts: int = 120
    seed: int = 0


@dataclass(frozen=True)
class PlannerResult:
    ok: bool
    path: tuple[tuple[float, ...], ...]
    reason: str
    nodes: int
    elapsed_s: float
    raw_waypoints: int = 0
    smoothed_waypoints: int = 0

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "joint_path": [list(point) for point in self.path],
            "reason": self.reason,
            "nodes": self.nodes,
            "elapsed_s": round(self.elapsed_s, 3),
            "raw_waypoints": self.raw_waypoints,
            "smoothed_waypoints": self.smoothed_waypoints,
        }


def _interpolate(a: np.ndarray, b: np.ndarray, step: float) -> list[np.ndarray]:
    count = max(1, int(np.ceil(float(np.max(np.abs(b - a))) / step)))
    return [a + (b - a) * (index / count) for index in range(1, count + 1)]


def _default_edge_valid(
    state_valid: StatePredicate, edge_step: float,
) -> EdgePredicate:
    return lambda a, b: all(
        state_valid(point) for point in _interpolate(a, b, edge_step)
    )


def _trace(nodes: list[np.ndarray], parents: list[int], index: int) -> list[np.ndarray]:
    path = []
    while index >= 0:
        path.append(nodes[index])
        index = parents[index]
    path.reverse()
    return path


def _steer(source: np.ndarray, target: np.ndarray, step: float) -> np.ndarray:
    delta = target - source
    distance = float(np.linalg.norm(delta))
    if distance <= step:
        return target.copy()
    return source + delta * (step / max(1e-12, distance))


def _nearest(nodes: list[np.ndarray], target: np.ndarray, scale: np.ndarray) -> int:
    distances = [float(np.linalg.norm((node - target) / scale)) for node in nodes]
    return int(np.argmin(distances))


def _shortcut(
    path: list[np.ndarray], edge_valid: EdgePredicate,
    rng: np.random.Generator, attempts: int,
) -> list[np.ndarray]:
    shortened = list(path)
    for _ in range(max(0, attempts)):
        if len(shortened) <= 2:
            break
        first, second = sorted(rng.choice(len(shortened), size=2, replace=False))
        if second <= first + 1:
            continue
        if edge_valid(shortened[first], shortened[second]):
            shortened = shortened[:first + 1] + shortened[second:]
    return shortened


def plan_rrt_connect(
    start: np.ndarray,
    goal: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    state_valid: StatePredicate,
    *,
    edge_valid: EdgePredicate | None = None,
    settings: PlannerSettings | None = None,
) -> PlannerResult:
    """Find a bounded collision-free path and return a dense joint polyline."""
    cfg = settings or PlannerSettings()
    began = time.monotonic()
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if not (start.shape == goal.shape == lower.shape == upper.shape):
        return PlannerResult(False, (), "joint vector dimensions differ", 0, 0.0)
    if np.any(start < lower) or np.any(start > upper) \
            or np.any(goal < lower) or np.any(goal > upper):
        return PlannerResult(False, (), "start or goal violates joint limits", 0, 0.0)
    if not state_valid(start):
        return PlannerResult(False, (), "start configuration is in collision", 0, 0.0)
    if not state_valid(goal):
        return PlannerResult(False, (), "goal configuration is in collision", 0, 0.0)
    edge = edge_valid or _default_edge_valid(
        state_valid, max(0.005, float(cfg.edge_step_rad))
    )
    if edge(start, goal):
        dense = [start, *_interpolate(start, goal, cfg.edge_step_rad)]
        return PlannerResult(
            True, tuple(tuple(float(v) for v in q) for q in dense),
            "direct joint edge is collision-free", 2,
            time.monotonic() - began, 2, len(dense),
        )

    rng = np.random.default_rng(int(cfg.seed))
    scale = np.maximum(upper - lower, 1e-6)
    trees = [([start.copy()], [-1]), ([goal.copy()], [-1])]
    roots = ["start", "goal"]

    while sum(len(tree[0]) for tree in trees) < max(2, int(cfg.max_nodes)):
        if time.monotonic() - began >= max(0.05, float(cfg.timeout_s)):
            break
        sample = (
            trees[1][0][0].copy()
            if rng.random() < float(np.clip(cfg.goal_bias, 0.0, 0.5))
            else rng.uniform(lower, upper)
        )
        nodes_a, parents_a = trees[0]
        near_a = _nearest(nodes_a, sample, scale)
        new_a = _steer(nodes_a[near_a], sample, max(0.02, cfg.step_rad))
        if not state_valid(new_a) or not edge(nodes_a[near_a], new_a):
            trees.reverse()
            roots.reverse()
            continue
        nodes_a.append(new_a)
        parents_a.append(near_a)
        index_a = len(nodes_a) - 1

        nodes_b, parents_b = trees[1]
        near_b = _nearest(nodes_b, new_a, scale)
        index_b = near_b
        connected = False
        while True:
            new_b = _steer(nodes_b[index_b], new_a, max(0.02, cfg.step_rad))
            if not state_valid(new_b) or not edge(nodes_b[index_b], new_b):
                break
            nodes_b.append(new_b)
            parents_b.append(index_b)
            index_b = len(nodes_b) - 1
            if float(np.linalg.norm(new_b - new_a)) <= 1e-7:
                connected = True
                break
            if time.monotonic() - began >= max(0.05, float(cfg.timeout_s)):
                break
        if connected:
            path_a = _trace(nodes_a, parents_a, index_a)
            path_b = _trace(nodes_b, parents_b, index_b)
            if roots[0] == "start":
                raw = path_a + list(reversed(path_b[:-1]))
            else:
                raw = path_b + list(reversed(path_a[:-1]))
            smooth = _shortcut(raw, edge, rng, cfg.shortcut_attempts)
            dense = [smooth[0]]
            for left, right in zip(smooth[:-1], smooth[1:]):
                dense.extend(_interpolate(left, right, cfg.edge_step_rad))
            return PlannerResult(
                True, tuple(tuple(float(v) for v in q) for q in dense),
                "bounded bidirectional RRT-Connect found a certified candidate",
                sum(len(tree[0]) for tree in trees), time.monotonic() - began,
                len(raw), len(dense),
            )
        trees.reverse()
        roots.reverse()

    return PlannerResult(
        False, (), "bounded RRT-Connect exhausted its time/node budget",
        sum(len(tree[0]) for tree in trees), time.monotonic() - began,
    )
