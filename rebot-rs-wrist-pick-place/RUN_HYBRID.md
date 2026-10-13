# Hybrid Claw-reflex + Hermes wrist pipeline

This branch combines Hermes metric wrist RGB-D scene understanding and placement
with a direct, OBB-driven reflex grasp executor. It does not run the original
hard-coded Claw teddy/bottle macro.

1. Hermes detects the named object, builds a calibrated RGB-D 3-D OBB, and
   validates a bounded scene/collision model.
2. The reflex selects a reachable OBB axis/cross-section candidate and commits
   a prevalidated direct terminal approach without close-range semantic replan.
3. A camera/geometry micro-lift must verify rigid object motion with the wrist.
4. Hermes re-detects the destination and plans payload-aware carry/place from
   the measured object-to-EE transform. A side/diagonal reflex grasp is never
   treated as a nominal top-down grasp.

## Clone and install

```bash
git clone --branch hybrid-claw-reflex-2026-10-13 \
  https://github.com/manavgoel472003/hermes_skills.git
cd hermes_skills/rebot-rs-wrist-pick-place
python -m pip install -r requirements-hybrid.txt
bash scripts/bootstrap_controller.sh
python -m pip install -e reBotArm_control_py_fresh
```

The repository includes the tested controller source and RobStride/CAN
acknowledgement changes. The bootstrap only validates it before installation.
The branch deliberately excludes local calibration, weights, Hugging Face
tokens, `.env`, model caches, and outputs.
Create your own local `.env` with `HF_TOKEN=...` if SAM3 is enabled.

Generate this robot's `calib/` files using the TCP, jaw, camera mount, table,
and drop-zone procedures in [arm_agent/README.md](arm_agent/README.md). Consult
[CAN.md](CAN.md) before enabling hardware. All physical commands require
`--run`; keep the physical E-stop available.

## Commands

```bash
# No hardware motion: inspect OBB candidates.
bash run_claw_box_reflex_preview.sh "green marker" --candidate-ladder --live

# Primary direct OBB reflex: two fresh candidates, then one Hermes fallback.
bash run_direct_reflex_mission.sh "green marker" \
  --routine scene_place --relation stack --reference "white cup" --run --live

# Hermes measured pick/place only.
bash run_hybrid_mission.sh "green marker" \
  --routine scene_place --relation stack --reference "white cup" --run --live
```

For a stack task keep the intended cup on the arm work surface and inside the
guarded workspace. Remove similarly described objects from the wrist view;
placement refuses an ambiguous or unreachable reference rather than guessing.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_hybrid_mission.py
python -m py_compile arm_agent/wrist_servo.py arm_agent/hybrid_mission.py
```
