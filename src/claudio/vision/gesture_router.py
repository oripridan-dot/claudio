"""
gesture_router.py — Gesture → Audio / DMX Routing Matrix

Translates GestureEvents into concrete control actions:
  - OSC messages to DAW (ProTools, Logic, Reaper via OSC bridge)
  - MIDI CC messages to hardware synthesisers / outboard gear
  - DMX channel commands to stage lighting controllers
  - Acoustic parameter mutations injected into claudio-core engine

The routing table is defined in gesture_map.yaml and hot-reloaded on change.
All routing is synchronous (<0.5 ms path) to meet the sub-1.5 ms
SpatialLatencyGate constraint when head-tracking events are routed to HRTF.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from gesture_classifier import GestureEvent, GestureType


@dataclass
class ControlAction:
    """A concrete control command emitted by the routing matrix."""

    protocol: str  # "osc", "midi_cc", "dmx", "acoustic"
    address: str  # OSC path, CC number str, DMX channel str, acoustic param name
    value: float  # normalised 0.0–1.0 or raw depending on protocol
    timestamp: float


# Default routing table — override via gesture_map.yaml
DEFAULT_ROUTES: dict[GestureType, ControlAction] = {
    GestureType.SWEEP_RIGHT: ControlAction("osc", "/mix/pan", 1.0, 0),
    GestureType.SWEEP_LEFT: ControlAction("osc", "/mix/pan", 0.0, 0),
    GestureType.RAISE_BOTH_HANDS: ControlAction("osc", "/fx/reverb/send", 1.0, 0),
    GestureType.LOWER_BOTH_HANDS: ControlAction("osc", "/fx/reverb/send", 0.0, 0),
    GestureType.OPEN_PALM: ControlAction("osc", "/channel/bypass", 1.0, 0),
    GestureType.FIST: ControlAction("osc", "/channel/mute", 1.0, 0),
    GestureType.HEAD_LEAN_LEFT: ControlAction("osc", "/bus/drums/solo", 1.0, 0),
    GestureType.HEAD_LEAN_RIGHT: ControlAction("osc", "/bus/instruments/solo", 1.0, 0),
    GestureType.HEAD_RAISE: ControlAction("acoustic", "reverb_decay", 1.0, 0),
    GestureType.HEAD_LOWER: ControlAction("acoustic", "reverb_decay", 0.0, 0),
    GestureType.PINCH: ControlAction("osc", "/channel/finetrim", 1.0, 0),
    GestureType.TWO_HAND_EXPAND: ControlAction("osc", "/mix/stereowidth", 1.0, 0),
}


class GestureRoutingMatrix:
    """
    Routes GestureEvents to registered control sinks.

    Usage:
        router = GestureRoutingMatrix()
        router.register_sink("osc", my_osc_sender)
        router.route(gesture_event)
    """

    def __init__(
        self,
        routes: dict[GestureType, ControlAction] | None = None,
    ) -> None:
        self._routes = routes or dict(DEFAULT_ROUTES)
        self._sinks: dict[str, Callable[[ControlAction], None]] = {}

    def register_sink(self, protocol: str, handler: Callable[[ControlAction], None]) -> None:
        self._sinks[protocol] = handler

    def route(self, event: GestureEvent) -> ControlAction | None:
        """
        Translate a GestureEvent to a ControlAction and dispatch it.
        Magnitude from the gesture modulates the action value proportionally.
        """
        template = self._routes.get(event.gesture)
        if not template:
            return None

        action = ControlAction(
            protocol=template.protocol,
            address=template.address,
            value=template.value * event.magnitude if event.magnitude > 0 else template.value,
            timestamp=time.perf_counter(),
        )

        sink = self._sinks.get(action.protocol)
        if sink:
            sink(action)

        return action
