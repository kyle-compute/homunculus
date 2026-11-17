"""What the runtime keeps for readers: issued observations, image bytes, and events.

Each store is bounded. An ID that fell out of retention is reported as expired,
which is different from an ID that never existed. The runtime calls this under
its own lock; nothing here knows about sessions, goals, or the backend.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .models import Event, ImageHandle, Observation, ResourceStatus, RobotError, Sample, ServiceLimits


@dataclass(frozen=True)
class Issued:
    """An observation plus what is needed to judge it later as evidence."""

    observation: Observation
    issued_at: float  # host clock
    measured_at: float  # host clock when its samples were read from the backend
    state_sequence: int  # which backend reading it shows


class History:
    def __init__(self, robot_id: str, boot_id: str, limits: ServiceLimits):
        self._robot_id = robot_id
        # IDs name one robot in one boot. Robots of a service share its boot, so the robot is part of the tag:
        # an observation of one robot can never be found, as evidence, under the same ID on another.
        self._tag = hashlib.sha256(f"{boot_id}:{robot_id}".encode()).hexdigest()[:8]
        self._limits = limits
        self._observations: OrderedDict[str, Issued] = OrderedDict()
        self._observation_count = 0
        self._images: OrderedDict[str, tuple[str, bytes]] = OrderedDict()
        self._image_bytes = 0
        self._image_count = 0
        self._events: deque[Event] = deque(maxlen=limits.event_retention)
        self._event_count = 0

    def issue(
        self,
        *,
        context_id: str,
        now: float,
        measured_at: float,
        state_sequence: int,
        samples: Mapping[str, Sample],
        resources: Sequence[ResourceStatus],
        images: Sequence[ImageHandle] = (),
    ) -> Observation:
        """Record a new immutable observation. Old samples keep their own capture times."""
        self._observation_count += 1
        observation = Observation(
            observation_id=f"obs-{self._tag}-{self._observation_count}",
            robot_id=self._robot_id,
            context_id=context_id,
            sequence=self._observation_count,
            issued_time_ns=str(round(now * 1e9)),
            samples=dict(samples),
            resources=list(resources),
            images=list(images),
        )
        self._observations[observation.observation_id] = Issued(observation, now, measured_at, state_sequence)
        while len(self._observations) > self._limits.observation_retention:
            self._observations.popitem(last=False)
        return observation

    def find(self, observation_id: str) -> Issued | None:
        """A retained observation, for use as a basis or start condition."""
        return self._observations.get(observation_id)

    def with_sequence(self, sequence: int) -> Issued | None:
        return next((issued for issued in self._observations.values() if issued.observation.sequence == sequence), None)

    def observation(self, observation_id: str) -> Observation:
        issued = self._observations.get(observation_id)
        if issued is not None:
            return issued.observation
        if self._was_issued(observation_id, f"obs-{self._tag}", self._observation_count):
            raise RobotError(
                "observation_expired",
                f"observation {observation_id} is no longer retained",
                retention=self._limits.observation_retention,
            )
        raise RobotError("not_found", f"no observation with id {observation_id}")

    def add_image(self, media_type: str, data: bytes) -> str:
        """Keep image bytes as given and return their handle. The oldest are dropped at the byte budget."""
        self._image_count += 1
        handle = f"img-{self._tag}-{self._image_count}"
        self._images[handle] = (media_type, data)
        self._image_bytes += len(data)
        while self._image_bytes > self._limits.image_store_bytes and len(self._images) > 1:
            _, (_, dropped) = self._images.popitem(last=False)
            self._image_bytes -= len(dropped)
        return handle

    def image(self, handle: str) -> tuple[str, bytes]:
        if handle in self._images:
            return self._images[handle]
        if self._was_issued(handle, f"img-{self._tag}", self._image_count):
            raise RobotError("image_expired", f"image {handle} is no longer retained")
        raise RobotError("not_found", f"no image with handle {handle}")

    def emit(self, event: str, now: float, /, **data: Any) -> None:
        self._event_count += 1
        self._events.append(Event(sequence=self._event_count, time_ns=str(round(now * 1e9)), type=event, data=data))

    def events(self, after: int) -> tuple[list[Event], bool]:
        """Events numbered above `after`, and whether the reader has a gap before them.

        Numbering starts at 1 in each boot. A cursor above the newest event was read in an earlier boot, so
        the reader gets this boot's events from the first one, as a gap.
        """
        earlier_boot = after > self._event_count
        if earlier_boot:
            after = 0
        newer = [event for event in self._events if event.sequence > after]
        oldest = self._events[0].sequence if self._events else self._event_count + 1
        return newer, earlier_boot or after + 1 < oldest

    @staticmethod
    def _was_issued(identifier: str, prefix: str, count: int) -> bool:
        head, _, number = identifier.rpartition("-")
        # ASCII digits only: other characters that count as digits, such as "²", are not numbers to int().
        return head == prefix and number.isascii() and number.isdigit() and 0 < int(number) <= count
