"""
Teleoperation Data Recorder

Records demonstration episodes from the ESP32 robot arm with synchronized
camera observations. Outputs data in HDF5 format compatible with LeRobot.

Usage:
    python -m host.data_collection.teleop_recorder \
        --task pick_place \
        --episodes 50 \
        --save_dir datasets/pick_place_v1
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import h5py
import cv2
import websockets
import requests
from threading import Thread, Event
from queue import Queue
import argparse


@dataclass
class Observation:
    """Single timestep observation."""
    timestamp_ms: int
    image: np.ndarray  # (H, W, C) RGB
    joint_positions: np.ndarray  # (4,) joint angles in degrees
    gripper_state: float  # 0.0 = open, 1.0 = closed


@dataclass
class Action:
    """Single timestep action (teleop input)."""
    timestamp_ms: int
    joint_targets: np.ndarray  # (4,) target joint angles
    teleop_input: np.ndarray  # (4,) raw joystick values


@dataclass
class Episode:
    """Complete episode recording."""
    task_name: str
    episode_id: int
    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    success: bool = False


class CameraClient:
    """MJPEG stream client for ESP32-CAM."""

    def __init__(self, url: str):
        self.url = url
        self.frame_queue = Queue(maxsize=2)
        self.running = Event()
        self.thread: Optional[Thread] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_timestamp: int = 0

    def start(self):
        """Start the camera stream reader thread."""
        self.running.set()
        self.thread = Thread(target=self._stream_reader, daemon=True)
        self.thread.start()
        print(f"Camera stream started: {self.url}")

    def stop(self):
        """Stop the camera stream."""
        self.running.clear()
        if self.thread:
            self.thread.join(timeout=2.0)

    def _stream_reader(self):
        """Background thread to read MJPEG stream."""
        stream = requests.get(f"{self.url}/stream", stream=True, timeout=10)
        bytes_buffer = b''

        for chunk in stream.iter_content(chunk_size=1024):
            if not self.running.is_set():
                break

            bytes_buffer += chunk
            # Find JPEG markers
            start = bytes_buffer.find(b'\xff\xd8')  # JPEG start
            end = bytes_buffer.find(b'\xff\xd9')    # JPEG end

            if start != -1 and end != -1 and end > start:
                jpg_bytes = bytes_buffer[start:end + 2]
                bytes_buffer = bytes_buffer[end + 2:]

                # Decode JPEG
                img_array = np.frombuffer(jpg_bytes, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if frame is not None:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.latest_frame = frame
                    self.latest_timestamp = int(time.time() * 1000)

    def get_frame(self) -> tuple[np.ndarray, int]:
        """Get the latest frame and timestamp."""
        if self.latest_frame is None:
            # Return placeholder if no frame yet
            return np.zeros((240, 320, 3), dtype=np.uint8), 0
        return self.latest_frame.copy(), self.latest_timestamp


class TeleopRecorder:
    """Records teleoperation demonstrations from ESP32 robot arm."""

    def __init__(
        self,
        arm_url: str = "ws://robot-arm.local:81",
        camera_url: str = "http://robot-cam.local",
        control_freq: int = 20
    ):
        self.arm_url = arm_url
        self.camera_url = camera_url
        self.control_freq = control_freq
        self.period = 1.0 / control_freq

        self.camera = CameraClient(camera_url)
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.current_state: dict = {}
        self.recording = False

    async def connect(self):
        """Connect to ESP32 arm and start camera stream."""
        print(f"Connecting to arm at {self.arm_url}...")
        self.websocket = await websockets.connect(self.arm_url)
        print("Arm connected!")

        self.camera.start()
        await asyncio.sleep(1.0)  # Wait for camera to stabilize

    async def disconnect(self):
        """Disconnect from robot."""
        if self.websocket:
            await self.websocket.close()
        self.camera.stop()

    async def _receive_state(self):
        """Background task to receive state updates."""
        async for message in self.websocket:
            data = json.loads(message)
            if data.get("type") == "state":
                self.current_state = data

    async def send_command(self, cmd: str, **kwargs):
        """Send command to robot."""
        message = {"cmd": cmd, **kwargs}
        await self.websocket.send(json.dumps(message))

    async def record_episode(
        self,
        task_name: str,
        episode_id: int,
        max_steps: int = 500,
        auto_start: bool = True
    ) -> Episode:
        """Record a single demonstration episode."""

        episode = Episode(task_name=task_name, episode_id=episode_id)

        print(f"\n=== Recording Episode {episode_id} for '{task_name}' ===")
        print("Use joysticks to control the robot.")
        print("Press joystick button to end episode (or wait for max steps).")

        if auto_start:
            input("Press ENTER to start recording...")

        # Start recording mode on ESP32
        await self.send_command("start_recording")
        self.recording = True
        episode.start_time = time.time()

        # Start receiving state in background
        receive_task = asyncio.create_task(self._receive_state())

        try:
            step = 0
            while step < max_steps and self.recording:
                step_start = time.time()

                # Wait for state update
                await asyncio.sleep(0.01)

                if not self.current_state:
                    continue

                # Capture synchronized observation
                frame, cam_timestamp = self.camera.get_frame()

                obs = Observation(
                    timestamp_ms=self.current_state.get("timestamp_ms", 0),
                    image=frame,
                    joint_positions=np.array(
                        self.current_state.get("joint_positions", [0, 0, 0, 0]),
                        dtype=np.float32
                    ),
                    gripper_state=self.current_state.get("gripper", 0.5)
                )
                episode.observations.append(obs)

                # Record action (teleop input as action)
                action = Action(
                    timestamp_ms=self.current_state.get("timestamp_ms", 0),
                    joint_targets=np.array(
                        self.current_state.get("joint_targets", [90, 90, 90, 45]),
                        dtype=np.float32
                    ),
                    teleop_input=np.array(
                        self.current_state.get("teleop_input", [0, 0, 0, 0]),
                        dtype=np.float32
                    )
                )
                episode.actions.append(action)

                step += 1

                # Check for episode end (button press)
                # This would be signaled from ESP32 via state update

                # Maintain control frequency
                elapsed = time.time() - step_start
                if elapsed < self.period:
                    await asyncio.sleep(self.period - elapsed)

                # Progress indicator
                if step % 50 == 0:
                    print(f"  Step {step}/{max_steps}")

        finally:
            receive_task.cancel()
            await self.send_command("stop_recording")

        episode.end_time = time.time()
        print(f"Episode recorded: {len(episode.observations)} steps, "
              f"{episode.end_time - episode.start_time:.1f} seconds")

        # Ask for success label
        success_input = input("Was the episode successful? (y/n): ")
        episode.success = success_input.lower() == 'y'

        return episode

    def save_episode(self, episode: Episode, save_dir: Path):
        """Save episode to HDF5 file."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        filename = save_dir / f"episode_{episode.episode_id:04d}.hdf5"

        with h5py.File(filename, 'w') as f:
            # Metadata
            f.attrs['task_name'] = episode.task_name
            f.attrs['episode_id'] = episode.episode_id
            f.attrs['success'] = episode.success
            f.attrs['start_time'] = episode.start_time
            f.attrs['end_time'] = episode.end_time
            f.attrs['fps'] = self.control_freq
            f.attrs['num_steps'] = len(episode.observations)

            # Observations
            obs_group = f.create_group('observations')

            images = np.stack([o.image for o in episode.observations])
            obs_group.create_dataset(
                'images',
                data=images,
                compression='gzip',
                compression_opts=4
            )

            joint_pos = np.stack([o.joint_positions for o in episode.observations])
            obs_group.create_dataset('joint_positions', data=joint_pos)

            gripper = np.array([o.gripper_state for o in episode.observations])
            obs_group.create_dataset('gripper_state', data=gripper)

            timestamps = np.array([o.timestamp_ms for o in episode.observations])
            obs_group.create_dataset('timestamps', data=timestamps)

            # Actions
            actions_group = f.create_group('actions')

            joint_targets = np.stack([a.joint_targets for a in episode.actions])
            actions_group.create_dataset('joint_targets', data=joint_targets)

            teleop = np.stack([a.teleop_input for a in episode.actions])
            actions_group.create_dataset('teleop_input', data=teleop)

        print(f"Saved: {filename}")
        return filename

    async def collect_dataset(
        self,
        task_name: str,
        num_episodes: int,
        save_dir: str,
        max_steps_per_episode: int = 500
    ):
        """Collect a complete dataset of demonstrations."""
        save_path = Path(save_dir)

        print(f"\n{'='*50}")
        print(f"Collecting dataset: {task_name}")
        print(f"Episodes: {num_episodes}")
        print(f"Save directory: {save_path}")
        print(f"{'='*50}\n")

        await self.connect()

        try:
            for ep_id in range(num_episodes):
                print(f"\n--- Episode {ep_id + 1}/{num_episodes} ---")

                # Move to home position
                await self.send_command("home")
                await asyncio.sleep(2.0)

                # Record episode
                episode = await self.record_episode(
                    task_name=task_name,
                    episode_id=ep_id,
                    max_steps=max_steps_per_episode
                )

                # Save episode
                self.save_episode(episode, save_path)

                print(f"Progress: {ep_id + 1}/{num_episodes} episodes complete")

        finally:
            await self.disconnect()

        print(f"\n{'='*50}")
        print(f"Dataset collection complete!")
        print(f"Saved to: {save_path}")
        print(f"{'='*50}")


async def main():
    parser = argparse.ArgumentParser(description="Record teleoperation demonstrations")
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--save_dir", type=str, default="datasets", help="Save directory")
    parser.add_argument("--arm_url", type=str, default="ws://robot-arm.local:81")
    parser.add_argument("--camera_url", type=str, default="http://robot-cam.local")
    parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")

    args = parser.parse_args()

    recorder = TeleopRecorder(
        arm_url=args.arm_url,
        camera_url=args.camera_url
    )

    await recorder.collect_dataset(
        task_name=args.task,
        num_episodes=args.episodes,
        save_dir=f"{args.save_dir}/{args.task}",
        max_steps_per_episode=args.max_steps
    )


if __name__ == "__main__":
    asyncio.run(main())
