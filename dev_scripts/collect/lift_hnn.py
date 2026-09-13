#!/usr/bin/env python3
"""Exp 3 environments: "lift the cube to N cm", plus a goal marker that stays out of depth.

Robosuite's `Lift` hardcodes success as `cube_height > table_height + 0.04`
(`lift.py:417`), so a per-height task needs a subclass. mimicgen's
`Stack_D0._check_lifted(body_id, margin=)` (`stack.py:57`) is the precedent, and
robosuite's `EnvMeta` registers a subclass by name the moment it is defined --
nothing in robosuite or mimicgen is edited.

**The half-height trap.** The cube's half-height is ~0.021, so at rest its *origin*
already sits at `table_top + 0.021`; the stock `+0.04` is therefore only ~2 cm of
real lift. Success here is measured against where the cube actually started:

    cube_z - cube_z_at_reset > target_height

**The goal marker and depth.** A visual geom is rasterised into the same pass as
depth, so a marker drawn for the viewer would also land in `agentview_depth` and
therefore in `agentview_pointmap` -- the goal frames would differ from the plain
frames in *all three* modalities, not just RGB. The marker is put in its own geom
group, which `MjvOption.geomgroup` switches off before the scene is even built, so
the plain observation cannot see it in colour or in depth. `goal_marker_visible()`
turns it on for one extra RGB render of the *same* simulator state, which is what
the qualitative videos display.
"""

import numpy as np
from contextlib import contextmanager

from robosuite.environments.manipulation.lift import Lift
from robosuite.models.objects import CylinderObject

from mimicgen.env_interfaces.base import make_interface  # noqa: F401  (registry side effects)
from mimicgen.env_interfaces.robosuite import RobosuiteInterface

# Robosuite uses group 0 for collision geoms and 1 for visual geoms, so 2 is free.
GOAL_GEOM_GROUP = 2

TARGET_HEIGHTS_CM = (5, 10, 15, 20)


class Lift_HNN(Lift):
    """Lift the cube `target_height` metres above where it started."""

    target_height = 0.05  # metres; each concrete subclass overrides this

    def __init__(self, **kwargs):
        self._cube_reset_z = None
        super().__init__(**kwargs)

    # ------------------------------------------------------------------ model

    def _load_model(self):
        super()._load_model()
        self.goal_marker = CylinderObject(
            name="goal_marker",
            size=[0.06, 0.0015],  # radius, half-height: a thin disc at the target plane
            rgba=[0.15, 0.95, 0.25, 0.55],
            obj_type="visual",
            joints=None,
        )
        self.model.merge_assets(self.goal_marker)
        self.model.worldbody.append(self.goal_marker.get_obj())

    def _setup_references(self):
        super()._setup_references()
        self.goal_marker_body_id = self.sim.model.body_name2id(self.goal_marker.root_body)
        self.goal_marker_geom_ids = [self.sim.model.geom_name2id(g) for g in self.goal_marker.visual_geoms]
        for geom_id in self.goal_marker_geom_ids:
            self.sim.model.geom_group[geom_id] = GOAL_GEOM_GROUP
        self._hide_goal_marker()

    # ------------------------------------------------------------------ reset

    def _reset_internal(self):
        super()._reset_internal()
        # The placement sampler has just written the cube's qpos; forward() is what
        # turns that into the body_xpos this whole task measures against.
        self.sim.forward()
        self._cube_reset_z = float(self.sim.data.body_xpos[self.cube_body_id][2])

        marker_pos = np.array(self.sim.data.body_xpos[self.cube_body_id], dtype=np.float64)
        marker_pos[2] = self._cube_reset_z + self.target_height
        self.sim.model.body_pos[self.goal_marker_body_id] = marker_pos
        self.sim.forward()
        self._hide_goal_marker()

    # ------------------------------------------------- goal marker visibility

    def _set_goal_marker_visible(self, visible: bool) -> None:
        context = self.sim._render_context_offscreen
        if context is not None:
            context.vopt.geomgroup[GOAL_GEOM_GROUP] = 1 if visible else 0

    def _hide_goal_marker(self) -> None:
        self._set_goal_marker_visible(False)

    @contextmanager
    def goal_marker_visible(self):
        """Render the marker for the duration of the block, then hide it again."""
        self._set_goal_marker_visible(True)
        try:
            yield
        finally:
            self._set_goal_marker_visible(False)

    # ---------------------------------------------------------------- success

    def cube_lift_height(self) -> float:
        """Metres the cube has risen since reset. The half-height cancels out."""
        if self._cube_reset_z is None:
            return 0.0
        return float(self.sim.data.body_xpos[self.cube_body_id][2]) - self._cube_reset_z

    def _check_success(self):
        return bool(self.cube_lift_height() > self.target_height)


def _make_height_subclass(centimetres: int):
    name = f"Lift_H{centimetres}"
    return type(name, (Lift_HNN,), {"target_height": centimetres / 100.0, "__doc__": f"Lift the cube {centimetres} cm."})


# Defining these registers them with robosuite's EnvMeta, so `robosuite.make("Lift_H5")`
# and robomimic's env_meta round-trip both work with no further wiring.
Lift_H5, Lift_H10, Lift_H15, Lift_H20 = (_make_height_subclass(cm) for cm in TARGET_HEIGHTS_CM)

ENV_NAMES = {cm: f"Lift_H{cm}" for cm in TARGET_HEIGHTS_CM}


class MG_Lift(RobosuiteInterface):
    """Env interface for `Lift` and its height variants -- one object, one subtask."""

    def get_object_poses(self):
        return dict(cube=self.get_object_pose(obj_name=self.env.cube.root_body, obj_type="body"))

    def get_subtask_term_signals(self):
        return dict(
            grasp=int(self.env._check_grasp(gripper=self.env.robots[0].gripper, object_geoms=self.env.cube))
        )
