from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest import TestCase

from arm_agent.local_arm import LocalArm


class SafeShutdownTest(TestCase):
    def _local(self, events):
        local = object.__new__(LocalArm)
        local._lock = threading.RLock()
        local._arm = SimpleNamespace(disconnect=lambda: events.append("disable_disconnect"))
        local._controller = SimpleNamespace()
        local.freeze = lambda: events.append("freeze") or {"ok": True}
        return local

    def test_disconnect_parks_before_motor_disable(self):
        events = []
        local = self._local(events)
        local.home = lambda: events.append("home") or {"ok": True}

        result = local.disconnect()

        self.assertTrue(result["ok"])
        self.assertEqual(events, ["freeze", "home", "disable_disconnect"])
        self.assertIsNone(local._arm)

    def test_failed_home_refuses_motor_disable_and_keeps_controller(self):
        events = []
        local = self._local(events)
        local.home = lambda: events.append("home") or {
            "ok": False, "reason": "home_not_confirmed",
        }

        result = local.disconnect()

        self.assertFalse(result["ok"])
        self.assertTrue(result["holding"])
        self.assertEqual(events, ["freeze", "home"])
        self.assertIsNotNone(local._arm)

    def test_failed_freeze_refuses_home_and_motor_disable(self):
        events = []
        local = self._local(events)
        local.freeze = lambda: events.append("freeze") or {
            "ok": False, "reason": "still moving",
        }
        local.home = lambda: events.append("home") or {"ok": True}

        result = local.disconnect()

        self.assertFalse(result["ok"])
        self.assertTrue(result["holding"])
        self.assertEqual(events, ["freeze"])
        self.assertIsNotNone(local._arm)
