import unittest

from memory.models.room import Room, RoomMatcher
from memory.rooms.registry import RoomRegistry


class RoomRegistryTests(unittest.TestCase):
    def test_screen_and_camera_are_the_only_auto_rooms(self):
        """One room per capture source — not one per activity type or project."""
        registry = RoomRegistry()

        registry.ensure_daily()
        registry.route({"source": "screen", "activity_type": "reading"})
        registry.route({"source": "screen", "activity_type": "coding",
                        "project_id": "home-assistant-ai"})
        registry.route({"source": "camera", "activity_type": "other"})

        self.assertEqual(sorted(registry.rooms), ["camera", "daily", "screen"])

    def test_screen_events_route_to_the_screen_room(self):
        registry = RoomRegistry()

        routed = registry.route({"source": "screen", "activity_type": "coding",
                                 "application": "pycharm64.exe"})

        self.assertEqual(routed.room_id, "screen")
        self.assertEqual(routed.kind, "screen")

    def test_camera_events_route_to_the_camera_room(self):
        registry = RoomRegistry()

        routed = registry.route({"source": "camera", "application": "ipc-a22e-g"})

        self.assertEqual(routed.room_id, "camera")
        # 'camera' kind is what marks the home memory domain in graph queries.
        self.assertEqual(routed.kind, "camera")

    def test_mobile_camera_capture_is_still_a_camera(self):
        registry = RoomRegistry()

        self.assertEqual(
            registry.route({"source": "mobile_camera"}).room_id, "camera")
        self.assertEqual(
            registry.route({"source": "mobile_screen"}).room_id, "screen")

    def test_events_with_no_source_fall_back_to_screen(self):
        """Older events predate the field; screen is the safe default."""
        registry = RoomRegistry()

        self.assertEqual(registry.route({"activity_type": "reading"}).room_id,
                         "screen")

    def test_both_source_rooms_exist_after_one_observation(self):
        """Routing a screen event must not leave Cameras missing, and vice versa."""
        registry = RoomRegistry()

        registry.route({"source": "screen"})

        self.assertIn("screen", registry.rooms)
        self.assertNotIn("camera", registry.rooms)  # ensured by the camera source
        registry.ensure_source_room("camera")
        self.assertIn("camera", registry.rooms)

    def test_user_topic_room_beats_the_source_room(self):
        scripture = Room(
            room_id="topic:scripture",
            name="Scripture",
            kind="topic",
            auto=False,
            matcher=RoomMatcher(title_keywords=["bible"]),
        )
        registry = RoomRegistry([scripture])

        routed = registry.route({
            "source": "screen",
            "activity_type": "reading",
            "summary": "Reading a Bible commentary",
        })

        self.assertEqual(routed.room_id, "topic:scripture")

    def test_archived_room_is_not_considered_for_routing(self):
        archived = Room(
            room_id="topic:old",
            name="Old",
            kind="topic",
            archived=True,
            matcher=RoomMatcher(title_keywords=["special phrase"]),
        )
        registry = RoomRegistry([archived])

        routed = registry.route({"source": "screen", "summary": "special phrase"})

        self.assertEqual(routed.room_id, "screen")

    def test_source_rooms_are_pinned_so_hygiene_leaves_them_alone(self):
        registry = RoomRegistry()

        for source in ("screen", "camera"):
            self.assertTrue(registry.ensure_source_room(source).pinned)


if __name__ == "__main__":
    unittest.main()
