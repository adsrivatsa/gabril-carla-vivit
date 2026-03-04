"""
Self-contained CARLA rollout evaluation for ViViT models.
Runs simulations without any dependency on GABRIL-CARLA.
Requires: CARLA server running, carla Python package (from CARLA install).
"""

from __future__ import print_function

import math
import os
import xml.etree.ElementTree as ET
from collections import deque
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as Fn

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False
    carla = None  # type: ignore


def _vector_to_control(vector: np.ndarray):
    """Convert 7-dim action vector to carla.VehicleControl."""
    if not CARLA_AVAILABLE:
        raise RuntimeError("carla package not available")
    ctrl = carla.VehicleControl()
    ctrl.throttle = float(np.clip(vector[0], 0.0, 1.0))
    ctrl.steer = float(np.clip(vector[1], -1.0, 1.0))
    ctrl.brake = float(vector[2] > 0.8)
    ctrl.hand_brake = bool(vector[3] > 0.5)
    ctrl.reverse = bool(vector[4] > 0.5)
    ctrl.manual_gear_shift = bool(vector[5] > 0.5)
    ctrl.gear = int(vector[6])
    return ctrl


def _noop_control():
    """Return no-op control (brake)."""
    if not CARLA_AVAILABLE:
        raise RuntimeError("carla package not available")
    ctrl = carla.VehicleControl()
    ctrl.throttle = 0.0
    ctrl.steer = 0.0
    ctrl.brake = 1.0
    ctrl.hand_brake = False
    ctrl.reverse = False
    ctrl.manual_gear_shift = False
    ctrl.gear = 0
    return ctrl


def _parse_route(
    routes_file: str, route_id: str
) -> Tuple[str, List[Tuple[float, float, float]]]:
    """
    Parse route from XML. Returns (town, waypoints) where waypoints are (x,y,z).
    """
    tree = ET.parse(routes_file)
    for route in tree.iter("route"):
        if route.attrib.get("id") != str(route_id):
            continue
        town = route.attrib["town"]
        waypoints = []
        wp_elem = route.find("waypoints")
        if wp_elem is not None:
            for pos in wp_elem.iter("position"):
                waypoints.append(
                    (
                        float(pos.attrib["x"]),
                        float(pos.attrib["y"]),
                        float(pos.attrib["z"]),
                    )
                )
        return town, waypoints
    raise ValueError(f"Route id {route_id} not found in {routes_file}")


def _dist_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _cls_attn_to_gaze_map(
    cls_attn: torch.Tensor,
    patch_size: Tuple[int, int],
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """
    Convert cls_attn (B, F, Heads, T) to spatial gaze map (H, W).
    T = (H//ph) * (W//pw). Reshape, interpolate to target size.
    """
    if cls_attn is None:
        return np.zeros((target_h, target_w), dtype=np.float32)
    # Mean over heads and frames
    attn = cls_attn.mean(dim=(1, 2))  # (B, T)
    attn = attn[0]  # (T,)
    ph, pw = patch_size
    pH = target_h // ph
    pW = target_w // pw
    attn = attn[: pH * pW].reshape(pH, pW)
    attn = attn.unsqueeze(0).unsqueeze(0)  # (1, 1, pH, pW)
    attn = Fn.interpolate(
        attn.float(),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )
    return attn[0, 0].cpu().numpy().astype(np.float32)


def run_single_rollout(
    model: torch.nn.Module,
    routes_file: str,
    route_id: str,
    host: str = "localhost",
    port: int = 2000,
    traffic_manager_port: int = 8000,
    seed: int = 199,
    frame_rate: float = 20.0,
    obs_res: Tuple[int, int] = (180, 320),
    frame_stack: int = 4,
    act_dim: int = 7,
    patch_size: Tuple[int, int] = (10, 10),
    max_steps: int = 2000,
    waypoint_threshold: float = 3.0,
    noop_steps: int = 10,
) -> Dict[str, Any]:
    """
    Run a single CARLA rollout with the given ViViT model.

    :param model: ViViT model (callable: obs -> (pred_a, cls_attn))
    :return: Dict with score_composed, score_route, score_penalty, num_collisions,
             route_length, steps, obs_frames, gaze_frames (for best rollout visualization)
    """
    H, W = obs_res
    device = next(model.parameters()).device

    if not CARLA_AVAILABLE:
        return {
            "score_composed": 0.0,
            "score_route": 0.0,
            "score_penalty": 1.0,
            "num_collisions": 0,
            "route_length": 0.0,
            "steps": 0,
            "obs_frames": [],
            "gaze_frames": [],
            "error": "carla package not available",
        }

    if not routes_file or not os.path.exists(routes_file):
        return {
            "score_composed": 0.0,
            "score_route": 0.0,
            "score_penalty": 1.0,
            "num_collisions": 0,
            "route_length": 0.0,
            "steps": 0,
            "obs_frames": [],
            "gaze_frames": [],
            "error": f"routes file not found: {routes_file}",
        }

    try:
        town, waypoints = _parse_route(routes_file, route_id)
    except (ValueError, ET.ParseError) as e:
        return {
            "score_composed": 0.0,
            "score_route": 0.0,
            "score_penalty": 1.0,
            "num_collisions": 0,
            "route_length": 0.0,
            "steps": 0,
            "obs_frames": [],
            "gaze_frames": [],
            "error": str(e),
        }

    client = None
    world = None
    actors: List[Any] = []

    try:
        client = carla.Client(host, port)
        client.set_timeout(10.0)
        world = client.load_world(town, reset_settings=False)
        world.set_weather(carla.WeatherParameters.ClearNoon)

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / frame_rate
        world.apply_settings(settings)

        tm = client.get_trafficmanager(traffic_manager_port)
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(seed)

        blueprint_lib = world.get_blueprint_library()
        vehicle_bp = blueprint_lib.filter("vehicle.tesla.model3")[0]
        spawn_point = carla.Transform(
            carla.Location(
                waypoints[0][0], waypoints[0][1], waypoints[0][2] + 1.0
            ),
            carla.Rotation(0, 0, 0),
        )
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actors.append(vehicle)

        cam_bp = blueprint_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(W))
        cam_bp.set_attribute("image_size_y", str(H))
        cam_bp.set_attribute("fov", "60")
        cam_transform = carla.Transform(
            carla.Location(x=0.7, z=1.60),
            carla.Rotation(),
        )
        camera = world.spawn_actor(
            cam_bp, cam_transform, attach_to=vehicle
        )
        actors.append(camera)

        collision_bp = blueprint_lib.find("sensor.other.collision")
        collision_sensor = world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=vehicle
        )
        actors.append(collision_sensor)

        collision_count = 0

        def _on_collision(_):
            nonlocal collision_count
            collision_count += 1

        collision_sensor.listen(_on_collision)

        image_queue: deque = deque(maxlen=1)

        def _on_image(im):
            arr = np.frombuffer(im.raw_data, dtype=np.uint8)
            arr = arr.reshape(H, W, 4)[:, :, :3]
            image_queue.append(arr)

        camera.listen(_on_image)

        route_length = 0.0
        for i in range(len(waypoints) - 1):
            x1, y1, _ = waypoints[i]
            x2, y2, _ = waypoints[i + 1]
            route_length += _dist_2d(x1, y1, x2, y2)

        frames_stack: deque = deque(maxlen=frame_stack)
        wp_idx = 0
        steps = 0

        obs_frames: List[np.ndarray] = []
        gaze_frames: List[np.ndarray] = []

        while steps < max_steps:
            world.tick()
            if not image_queue:
                continue

            img = image_queue[-1][:, :, ::-1].copy()  # BGR -> RGB
            if len(frames_stack) == 0:
                for _ in range(frame_stack):
                    frames_stack.append(img.copy())
            else:
                frames_stack.append(img.copy())

            stacked = np.stack(list(frames_stack), axis=0)
            stacked = torch.from_numpy(stacked).float() / 255.0
            stacked = stacked.permute(0, 3, 1, 2).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_a, cls_attn = model(stacked)
                v = pred_a[0].cpu().numpy()

            if len(v) < act_dim:
                v = np.pad(v, (0, act_dim - len(v)), mode="constant", constant_values=0)

            if steps < noop_steps:
                control = _noop_control()
            else:
                control = _vector_to_control(v.astype(np.float32))

            vehicle.apply_control(control)
            steps += 1

            if steps >= noop_steps and cls_attn is not None:
                obs_frames.append(img.copy())
                gaze_map = _cls_attn_to_gaze_map(
                    cls_attn, patch_size, H, W
                )
                gaze_frames.append(gaze_map)

            loc = vehicle.get_location()
            while wp_idx < len(waypoints):
                wx, wy, wz = waypoints[wp_idx]
                d = _dist_2d(loc.x, loc.y, wx, wy)
                if d <= waypoint_threshold:
                    wp_idx += 1
                else:
                    break

            if wp_idx >= len(waypoints):
                break

        route_completion = 100.0 * wp_idx / len(waypoints) if waypoints else 0.0
        score_route = min(route_completion, 100.0)

        penalty = 1.0
        if collision_count > 0:
            penalty *= 0.6 ** collision_count
        score_penalty = max(0.0, min(1.0, penalty))
        score_composed = score_route * score_penalty

        return {
            "score_composed": score_composed,
            "score_route": score_route / 100.0,
            "score_penalty": score_penalty,
            "num_collisions": collision_count,
            "route_length": route_length,
            "steps": steps,
            "waypoints_reached": wp_idx,
            "total_waypoints": len(waypoints),
            "obs_frames": obs_frames,
            "gaze_frames": gaze_frames,
        }

    except Exception as e:
        return {
            "score_composed": 0.0,
            "score_route": 0.0,
            "score_penalty": 1.0,
            "num_collisions": 0,
            "route_length": 0.0,
            "steps": 0,
            "obs_frames": [],
            "gaze_frames": [],
            "error": str(e),
        }
    finally:
        if world and world.get_settings().synchronous_mode:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        for a in reversed(actors):
            try:
                a.destroy()
            except Exception:
                pass
