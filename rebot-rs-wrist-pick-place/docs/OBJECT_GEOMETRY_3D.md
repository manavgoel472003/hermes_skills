# Metric object geometry for scene pick/place

The scene workflow builds a persistent, support-plane-aligned 3D box for every
picked object. Semantic vision (SAM3 or Grounded SAM) decides which pixels are
the object; aligned RealSense depth supplies metric scale. The controller fuses
the stable pre-pick frames in the robot base frame, rejects depth-edge outliers,
fits an oriented XY footprint, and uses the calibrated table plane for the
hidden bottom face.

The resulting record contains:

- observed and conservative XYZ dimensions;
- object centre and oriented axes in the robot base frame;
- support-plane Z, uncertainty, point count, and frame count.

It remains attached to the held-object context after the fingers obscure the
camera. Therefore a close-up partial mask cannot shorten a marker or sponge in
the collision model. Gripper fit, attached-payload collision checks, alternate
route planning, and release height all consume the same persistent geometry.

For placement, a requested relation chooses object-centre XY. The controller
then computes the object's vertical half-extent at the carried orientation and
sets its bottom `REBOT_PLACE_RELEASE_GAP_M` above the calibrated support plane.
The default gap is 10 mm. The resulting end-effector pose and the complete
payload route must still pass IK, workspace, self-collision, table, and scene
collision checks.

## Optional SAM 3D Objects completion

Meta's SAM 3D Objects model can be hosted as a separate service and selected by
setting `REBOT_SAM3D_ENDPOINT`. The service accepts JSON containing
`image_jpeg_base64` and `mask_png_base64`, and returns
`{"vertices": [[x, y, z], ...]}`.

The learned mesh is only a shape prior. It may expand an occluded dimension but
can never shrink an extent measured by RGB-D; incompatible or oversized priors
are rejected. With no endpoint—or if the service fails—the metric RGB-D box is
still used. Keeping the model outside the arm process also prevents a large
vision model from starving camera capture or motor safety handling.
The advisory request is bounded to 12 seconds by default
(`REBOT_SAM3D_TIMEOUT_S`, clamped to 2–30 seconds), after which planning falls
back to the measured RGB-D box.

## Operator-visible logs

Before the robot approaches, expect entries like:

```text
stage=object_box_3d ok=True source=metric_rgbd views=2 points=... observed_mm=[...] conservative_mm=[...] prior=...
stage=attach_payload_collision_envelope ... size_mm=[...] center_offset_ee_mm=[...]
stage=support_plane_release_pose ok=True support_z_m=... object_center=[...] ee_endpoint=[...]
```

If metric geometry or the support plane is unavailable, a new pick/place run is
refused before release planning rather than reverting to a guessed object size.
