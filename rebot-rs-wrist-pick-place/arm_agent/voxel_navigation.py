"""Top-down voxel navigation for the post-grasp carry phase.

This planner is deliberately only a route proposer.  It turns the measured
scene columns into an XY grid, searches free cells at several measured Z
layers, and returns sparse Cartesian anchors.  The arm controller must still
run its complete URDF IK and collision checks before executing them.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any

import numpy as np

from arm_agent.scene_collision import SceneMap


@dataclass(frozen=True)
class VoxelRoute:
    anchors_m: tuple[tuple[float, float, float], ...]
    release_endpoint_m: tuple[float, float, float]
    detail: dict[str, Any]


def _box_bounds(box: Any) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(box.center_m, dtype=float)
    half = np.asarray(box.size_m, dtype=float) / 2.0
    return center - half, center + half


def plan_topdown_route(
    scene: SceneMap, start_m: np.ndarray, destination_xy_m: np.ndarray, *,
    workspace_lo_m: tuple[float, float, float],
    workspace_hi_m: tuple[float, float, float],
    workspace_margin_m: float,
    ee_radius_m: float,
    vertical_clearance_m: float,
    drop_clearance_m: float,
    resolution_m: float,
    destination_radius_m: float | None = None,
) -> tuple[VoxelRoute | None, dict[str, Any]]:
    """Plan XY motion through free cells, trying higher Z layers as needed."""
    start = np.asarray(start_m, dtype=float)
    goal_xy = np.asarray(destination_xy_m, dtype=float)
    lo = np.asarray(workspace_lo_m, dtype=float) + workspace_margin_m
    hi = np.asarray(workspace_hi_m, dtype=float) - workspace_margin_m
    resolution = float(np.clip(resolution_m, 0.008, 0.030))
    if np.any(goal_xy < lo[:2]) or np.any(goal_xy > hi[:2]):
        return None, {"reason": "destination XY is outside the guarded workspace"}

    nx = max(2, int(math.floor((hi[0] - lo[0]) / resolution)) + 1)
    ny = max(2, int(math.floor((hi[1] - lo[1]) / resolution)) + 1)
    cell_tops = np.full((nx, ny), float(scene.table_z_m), dtype=float)
    inflate_cells = int(math.ceil(max(0.0, ee_radius_m) / resolution))

    # Rasterize the actual conservative collision boxes. Their measured height
    # is retained, allowing the search to go over a low obstacle or around a
    # tall one. XY expansion models the open gripper/end-effector footprint.
    for box in scene.boxes:
        box_lo, box_hi = _box_bounds(box)
        ix0 = max(0, int(math.floor((box_lo[0] - lo[0]) / resolution)) - inflate_cells)
        iy0 = max(0, int(math.floor((box_lo[1] - lo[1]) / resolution)) - inflate_cells)
        ix1 = min(nx - 1, int(math.ceil((box_hi[0] - lo[0]) / resolution)) + inflate_cells)
        iy1 = min(ny - 1, int(math.ceil((box_hi[1] - lo[1]) / resolution)) + inflate_cells)
        if ix0 <= ix1 and iy0 <= iy1:
            cell_tops[ix0:ix1 + 1, iy0:iy1 + 1] = np.maximum(
                cell_tops[ix0:ix1 + 1, iy0:iy1 + 1], float(box_hi[2])
            )

    def cell(xy: np.ndarray) -> tuple[int, int]:
        raw = np.rint((xy - lo[:2]) / resolution).astype(int)
        return int(np.clip(raw[0], 0, nx - 1)), int(np.clip(raw[1], 0, ny - 1))

    def world(node: tuple[int, int]) -> np.ndarray:
        return lo[:2] + resolution * np.asarray(node, dtype=float)

    start_cell, goal_cell = cell(start[:2]), cell(goal_xy)
    # Destination height must come from actual boxes overlapping the physical
    # payload/EE footprint.  Using the route grid here double-counted obstacle
    # inflation; arm-aware route retries then made nearby columns appear to sit
    # at the destination and falsely pushed release Z outside the workspace.
    destination_radius = max(
        0.0,
        float(ee_radius_m if destination_radius_m is None
              else destination_radius_m),
    )
    goal_top = float(scene.table_z_m)
    for box in scene.boxes:
        box_lo, box_hi = _box_bounds(box)
        closest_xy = np.clip(goal_xy, box_lo[:2], box_hi[:2])
        if float(np.linalg.norm(goal_xy - closest_xy)) \
                <= destination_radius + 1e-9:
            goal_top = max(goal_top, float(box_hi[2]))
    required_release_z = float(max(lo[2], goal_top + drop_clearance_m))
    if required_release_z > hi[2] + 1e-9:
        return None, {
            "reason": "destination column plus carried-object drop clearance "
                      "exceeds the guarded Z workspace",
            "destination_column_top_m": round(goal_top, 4),
            "required_release_z_m": round(required_release_z, 4),
            "maximum_release_z_m": round(float(hi[2]), 4),
        }
    release_z = required_release_z

    obstacle_tops = sorted({
        round(float(value + vertical_clearance_m), 4)
        for value in cell_tops.ravel()
        if value > scene.table_z_m + 0.003
    })
    # Prefer the already payload-aware release height for the whole traverse.
    # Previously the layers were sorted low-to-high, so a mathematically valid
    # path at the current carry height won even when a short vertical lift
    # would have cleared thin tabletop clutter (including geometry that is too
    # sparse to survive scene-map consolidation).  At this preferred layer A*
    # still routes in XY around columns that are too tall to pass over.
    #
    # Higher layers remain available for mapped barriers.  Lower layers are
    # retained last as a compatibility fallback when the preferred/high layers
    # have no grid route; complete-arm IK/collision validation remains the
    # controller's responsibility.
    base_z = max(float(start[2]), float(lo[2]))
    preferred_z = min(float(hi[2]), max(base_z, release_z))
    layer_epsilon = max(0.003, 0.25 * resolution)
    higher_layers = [
        min(float(hi[2]), z + layer_epsilon)
        for z in obstacle_tops
        if z > preferred_z + 0.002
    ]
    if higher_layers:
        higher_layers.append(
            min(float(hi[2]), max(higher_layers) + 0.020)
        )
    lower_fallbacks = [base_z]
    lower_fallbacks.extend(
        min(float(hi[2]), z + layer_epsilon)
        for z in obstacle_tops
        if base_z + 0.002 < z < preferred_z - 0.002
    )

    # De-duplicate without sorting: route priority is safety policy, not Z
    # magnitude.  Preferred payload-clear travel comes first, then higher
    # escape layers, then the legacy low-layer alternatives.
    layers: list[float] = []
    for candidate in [preferred_z, *higher_layers, *lower_fallbacks]:
        bounded = round(min(float(hi[2]), max(float(lo[2]), candidate)), 4)
        if bounded not in layers:
            layers.append(bounded)

    neighbors = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
    ]

    def blocked_at(layer_z: float) -> np.ndarray:
        blocked = cell_tops + vertical_clearance_m >= layer_z - 1e-6
        # The arm is physically at start, so map/self-filter discretization
        # cannot invalidate that one measured cell.
        blocked[start_cell] = False
        return blocked

    def search(
        layer_z: float, target_cell: tuple[int, int], *,
        allow_occupied_target: bool,
    ) -> list[tuple[int, int]] | None:
        blocked = blocked_at(layer_z)
        if blocked[target_cell] and not allow_occupied_target:
            return None
        if allow_occupied_target:
            blocked[target_cell] = False
        queue: list[tuple[float, float, tuple[int, int]]] = []
        heapq.heappush(queue, (0.0, 0.0, start_cell))
        previous: dict[tuple[int, int], tuple[int, int]] = {}
        cost = {start_cell: 0.0}
        while queue:
            _, current_cost, current = heapq.heappop(queue)
            if current == target_cell:
                path = [current]
                while current in previous:
                    current = previous[current]
                    path.append(current)
                return list(reversed(path))
            if current_cost > cost.get(current, float("inf")) + 1e-9:
                continue
            for dx, dy, step_cost in neighbors:
                nxt = (current[0] + dx, current[1] + dy)
                if not (0 <= nxt[0] < nx and 0 <= nxt[1] < ny):
                    continue
                if blocked[nxt]:
                    continue
                # Do not cut diagonally between two occupied columns.
                if dx and dy and (blocked[current[0] + dx, current[1]] or
                                  blocked[current[0], current[1] + dy]):
                    continue
                crowd = 0
                for ox in (-1, 0, 1):
                    for oy in (-1, 0, 1):
                        px, py = nxt[0] + ox, nxt[1] + oy
                        if 0 <= px < nx and 0 <= py < ny and blocked[px, py]:
                            crowd += 1
                new_cost = current_cost + step_cost + 0.08 * crowd
                if new_cost >= cost.get(nxt, float("inf")):
                    continue
                cost[nxt] = new_cost
                previous[nxt] = current
                heuristic = math.hypot(
                    nxt[0] - target_cell[0], nxt[1] - target_cell[1]
                )
                heapq.heappush(queue, (new_cost + heuristic, new_cost, nxt))
        return None

    attempted: list[dict[str, Any]] = []
    for layer_z in layers:
        rise_before_goal = layer_z < release_z - 1e-6
        staging_cell: tuple[int, int] | None = None
        if rise_before_goal:
            # At the lower carry layer the destination's own inflated object
            # footprint is intentionally blocked. Find the nearest reachable
            # free cell around that footprint, rise there, then cross above it
            # at release_z. This is what makes XY detours possible instead of
            # forcing the entire traverse to the destination release height.
            blocked = blocked_at(layer_z)
            candidates = [
                (ix, iy) for ix in range(nx) for iy in range(ny)
                if not blocked[ix, iy]
            ]
            candidates.sort(key=lambda node: (
                math.hypot(node[0] - goal_cell[0], node[1] - goal_cell[1]),
                math.hypot(node[0] - start_cell[0], node[1] - start_cell[1]),
            ))
            path = None
            # A small grid is used in practice; cap candidate probes so a
            # malformed map cannot turn planning into an unbounded search.
            for candidate in candidates[:64]:
                candidate_path = search(
                    layer_z, candidate, allow_occupied_target=False
                )
                if candidate_path is not None:
                    path = candidate_path
                    staging_cell = candidate
                    break
        else:
            path = search(
                layer_z, goal_cell, allow_occupied_target=True
            )
        attempted.append({"travel_z_m": layer_z, "found": path is not None})
        if path is None:
            continue
        # If the destination requires a higher release than the selected XY
        # travel layer, stop in the last free grid cell, rise there, and only
        # then cross above the destination.  Never enter an occupied goal
        # column low and climb through it.
        xy_path = path

        # Retain only turns. Exact destination XY replaces the quantized goal.
        sparse = [xy_path[0]]
        prior_direction = None
        for index in range(1, len(xy_path)):
            direction = (xy_path[index][0] - xy_path[index - 1][0],
                         xy_path[index][1] - xy_path[index - 1][1])
            if prior_direction is not None and direction != prior_direction:
                sparse.append(xy_path[index - 1])
            prior_direction = direction
        if sparse[-1] != xy_path[-1]:
            sparse.append(xy_path[-1])
        anchors: list[tuple[float, float, float]] = [
            (float(start[0]), float(start[1]), float(layer_z))
        ]
        if rise_before_goal:
            anchors.extend(
                (float(world(node)[0]), float(world(node)[1]), float(layer_z))
                for node in sparse[1:]
            )
            assert staging_cell is not None
            staging_xy = world(staging_cell)
            elevated_staging = (
                float(staging_xy[0]), float(staging_xy[1]), release_z
            )
            if anchors[-1] != elevated_staging:
                anchors.append(elevated_staging)
            anchors.append((
                float(goal_xy[0]), float(goal_xy[1]), release_z
            ))
        else:
            anchors.extend(
                (float(world(node)[0]), float(world(node)[1]), float(layer_z))
                for node in sparse[1:-1]
            )
            anchors.append((
                float(goal_xy[0]), float(goal_xy[1]), float(layer_z)
            ))
        endpoint = (float(goal_xy[0]), float(goal_xy[1]), release_z)
        if not rise_before_goal and abs(layer_z - release_z) > 1e-6:
            anchors.append(endpoint)
        rose_before_xy = layer_z > base_z + 0.002
        return VoxelRoute(tuple(anchors), endpoint, {
            "reason": "top-down voxel A* route found",
            "resolution_m": round(resolution, 4),
            "grid_cells": [nx, ny],
            "occupied_cells": int(np.count_nonzero(
                cell_tops > scene.table_z_m + 0.003
            )),
            "ee_radius_m": round(float(ee_radius_m), 4),
            "destination_radius_m": round(destination_radius, 4),
            "travel_z_m": round(float(layer_z), 4),
            "starting_carry_z_m": round(base_z, 4),
            "preferred_payload_clear_z_m": round(preferred_z, 4),
            "rose_before_xy": rose_before_xy,
            "destination_column_top_m": round(goal_top, 4),
            "release_z_m": round(release_z, 4),
            "path_cells": len(path) + (1 if rise_before_goal else 0),
            "path_turns": max(0, len(sparse) - 2),
            "travel_strategy": (
                "rise_before_xy_then_voxel_astar"
                if rose_before_xy else
                "xy_at_carry_height_then_rise_before_destination"
                if rise_before_goal else
                "xy_at_single_clear_layer"
            ),
            "route_anchors_m": [
                [round(value, 4) for value in anchor]
                for anchor in anchors
            ],
            "attempted_layers": attempted,
        }), {"reason": "ok", "attempted_layers": attempted}
    return None, {
        "reason": "no free top-down voxel route at any reachable Z layer",
        "attempted_layers": attempted,
        "grid_cells": [nx, ny],
    }
