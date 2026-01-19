"""
Real-time Policy Deployment

Runs a trained policy on the ESP32 robot arm with real-time
camera observations.

Usage:
    python -m host.deployment.realtime_control \
        --policy outputs/bc_policy_best.pt \
        --num_episodes 10
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import cv2
import websockets
import requests
from threading import Thread, Event
import argparse


class CameraClient:
    """MJPEG stream client for real-time inference."""

    def __init__(self, url: str):
        self.url = url
        self.running = Event()
        self.thread: Optional[Thread] = None
        self.latest_frame: Optional[np.ndarray] = None

    def start(self):
        self.running.set()
        self.thread = Thread(target=self._stream_reader, daemon=True)
        self.thread.start()

    def stop(self):
        self.running.clear()
        if self.thread:
            self.thread.join(timeout=2.0)

    def _stream_reader(self):
        stream = requests.get(f"{self.url}/stream", stream=True, timeout=10)
        bytes_buffer = b''
        for chunk in stream.iter_content(chunk_size=1024):
            if not self.running.is_set():
                break
            bytes_buffer += chunk
            start = bytes_buffer.find(b'\xff\xd8')
            end = bytes_buffer.find(b'\xff\xd9')
            if start != -1 and end != -1 and end > start:
                jpg_bytes = bytes_buffer[start:end + 2]
                bytes_buffer = bytes_buffer[end + 2:]
                img_array = np.frombuffer(jpg_bytes, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if frame is not None:
                    self.latest_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get_frame(self) -> np.ndarray:
        if self.latest_frame is None:
            return np.zeros((240, 320, 3), dtype=np.uint8)
        return self.latest_frame.copy()


class PolicyExecutor:
    """Executes a trained policy on the real robot."""

    def __init__(
        self,
        policy_path: str,
        arm_url: str = "ws://robot-arm.local:81",
        camera_url: str = "http://robot-cam.local",
        control_freq: int = 20,
        device: str = "cpu"
    ):
        self.arm_url = arm_url
        self.camera_url = camera_url
        self.control_freq = control_freq
        self.period = 1.0 / control_freq
        self.device = device
        self.policy = self._load_policy(policy_path)
        self.policy.eval()
        print(f"Policy loaded from {policy_path}")
        self.camera = CameraClient(camera_url)
        self.websocket = None
        self.current_state = {}

    def _load_policy(self, path: str):
        """Load trained policy from checkpoint."""
        from host.training.behavioral_cloning import BCPolicy, BCConfig
        checkpoint = torch.load(path, map_location=self.device)
        config = checkpoint.get('config', BCConfig())
        policy = BCPolicy(config).to(self.device)
        policy.load_state_dict(checkpoint['model_state_dict'])
        return policy

    async def connect(self):
        print(f"Connecting to arm at {self.arm_url}...")
        self.websocket = await websockets.connect(self.arm_url)
        await self.send_command("set_autonomous", value=True)
        self.camera.start()
        await asyncio.sleep(1.0)
        print("Connected!")

    async def disconnect(self):
        if self.websocket:
            await self.send_command("set_autonomous", value=False)
            await self.websocket.close()
        self.camera.stop()

    async def send_command(self, cmd: str, **kwargs):
        message = {"cmd": cmd, **kwargs}
        await self.websocket.send(json.dumps(message))

    async def get_state(self) -> dict:
        await self.send_command("get_state")
        response = await self.websocket.recv()
        return json.loads(response)

    async def send_action(self, action: np.ndarray):
        await self.send_command("set_targets", targets=action.tolist())

    def get_observation(self) -> dict:
        frame = self.camera.get_frame()
        state = np.array(
            self.current_state.get("joint_positions", [90, 90, 90, 45]) +
            [self.current_state.get("gripper", 0.5)],
            dtype=np.float32
        )
        return {"image": frame, "state": state}

    async def run_episode(self, max_steps: int = 200, record_video: bool = False) -> dict:
        """Run a single autonomous episode."""
        frames = [] if record_video else None
        inference_times = []

        print(f"Running episode (max {max_steps} steps)...")
        await self.send_command("home")
        await asyncio.sleep(2.0)

        step = 0
        episode_start = time.time()

        while step < max_steps:
            step_start = time.time()
            obs = self.get_observation()

            inference_start = time.time()
            with torch.no_grad():
                image_tensor = torch.from_numpy(obs["image"]).unsqueeze(0).to(self.device)
                state_tensor = torch.from_numpy(obs["state"]).unsqueeze(0).to(self.device)
                action = self.policy(image=image_tensor, state=state_tensor)
                action = action.cpu().numpy().squeeze()
            inference_times.append(time.time() - inference_start)

            await self.send_action(action)

            if record_video:
                frame = obs["image"].copy()
                cv2.putText(frame, f"Step: {step}", (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            try:
                self.current_state = await asyncio.wait_for(self.get_state(), timeout=0.1)
            except asyncio.TimeoutError:
                pass

            step += 1
            elapsed = time.time() - step_start
            if elapsed < self.period:
                await asyncio.sleep(self.period - elapsed)

            if step % 50 == 0:
                print(f"  Step {step}/{max_steps}")

        episode_duration = time.time() - episode_start

        if record_video and frames:
            video_path = f"episode_{int(time.time())}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, self.control_freq, (320, 240))
            for frame in frames:
                out.write(frame)
            out.release()
            print(f"Video saved: {video_path}")

        results = {
            "steps": step,
            "duration": episode_duration,
            "avg_inference_ms": np.mean(inference_times) * 1000,
            "control_freq_actual": step / episode_duration
        }
        print(f"Episode complete! Steps: {step}, Duration: {episode_duration:.1f}s")
        return results

    async def evaluate(self, num_episodes: int = 10, max_steps: int = 200) -> dict:
        """Evaluate policy over multiple episodes."""
        print(f"\nEvaluating policy over {num_episodes} episodes\n")
        await self.connect()

        all_results = []
        successes = 0

        try:
            for ep_idx in range(num_episodes):
                print(f"\n--- Episode {ep_idx + 1}/{num_episodes} ---")
                results = await self.run_episode(max_steps=max_steps)
                success_input = input("Successful? (y/n): ")
                results["success"] = success_input.lower() == 'y'
                if results["success"]:
                    successes += 1
                all_results.append(results)
        finally:
            await self.disconnect()

        success_rate = successes / num_episodes
        print(f"\nSuccess rate: {success_rate*100:.1f}%")
        return {"success_rate": success_rate, "results": all_results}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--arm_url", type=str, default="ws://robot-arm.local:81")
    parser.add_argument("--camera_url", type=str, default="http://robot-cam.local")
    parser.add_argument("--save_videos", action="store_true")
    args = parser.parse_args()

    executor = PolicyExecutor(
        policy_path=args.policy, arm_url=args.arm_url, camera_url=args.camera_url
    )

    if args.num_episodes == 1:
        await executor.connect()
        try:
            await executor.run_episode(max_steps=args.max_steps, record_video=args.save_videos)
        finally:
            await executor.disconnect()
    else:
        await executor.evaluate(num_episodes=args.num_episodes, max_steps=args.max_steps)


if __name__ == "__main__":
    asyncio.run(main())
