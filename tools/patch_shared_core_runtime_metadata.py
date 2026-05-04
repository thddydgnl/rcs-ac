from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "env" / "carla_env.py"


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"pattern not found:\n{old[:400]}")
    return text.replace(old, new, 1)


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")

    text = replace_once(
        text,
        """        self.last_target_speed_kmh = 0.0
        self.last_raw_longitudinal_action = 0.0
        self.last_action_smoothing_applied = False
""",
        """        self.last_target_speed_kmh = 0.0
        self.last_raw_longitudinal_action = 0.0
        self.last_speed_error_kmh = 0.0
        self.last_raw_requested_throttle = 0.0
        self.last_raw_requested_brake = 0.0
        self.last_smoothed_steer = 0.0
        self.last_smoothed_throttle = 0.0
        self.last_smoothed_brake = 0.0
        self.last_requested_applied_action_l1 = 0.0
        self.last_action_smoothing_applied = False
""",
    )

    # Reset blocks occur in multiple lifecycle paths.
    text = text.replace(
        """        self.last_target_speed_kmh = 0.0
        self.last_raw_longitudinal_action = 0.0
        self.last_action_smoothing_applied = False
        self.low_speed_steps = 0
""",
        """        self.last_target_speed_kmh = 0.0
        self.last_raw_longitudinal_action = 0.0
        self.last_speed_error_kmh = 0.0
        self.last_raw_requested_throttle = 0.0
        self.last_raw_requested_brake = 0.0
        self.last_smoothed_steer = 0.0
        self.last_smoothed_throttle = 0.0
        self.last_smoothed_brake = 0.0
        self.last_requested_applied_action_l1 = 0.0
        self.last_action_smoothing_applied = False
        self.low_speed_steps = 0
""",
    )

    text = replace_once(
        text,
        """    def step(self, action: Optional[Any] = None, **_: Any) -> Dict[str, Any]:
""",
        """    def step(
        self,
        action: Optional[Any] = None,
        policy_diagnostics: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> Dict[str, Any]:
""",
    )

    text = replace_once(
        text,
        """        base_info = self._build_info(
            frame=frame,
            current_step=current_step,
            sensor_snapshot=sensor_snapshot,
            action_vector=action_vector,
            control=control,
            route_metrics=route_metrics,
            observation=observation,
            reward=0.0,
            reward_breakdown=self._zero_reward_breakdown(),
            done=False,
            done_reason="running",
        )
""",
        """        base_info = self._build_info(
            frame=frame,
            current_step=current_step,
            sensor_snapshot=sensor_snapshot,
            action_vector=action_vector,
            control=control,
            route_metrics=route_metrics,
            observation=observation,
            reward=0.0,
            reward_breakdown=self._zero_reward_breakdown(),
            done=False,
            done_reason="running",
        )
        base_info["policy_diagnostics"] = dict(policy_diagnostics or {})
""",
    )

    text = replace_once(
        text,
        """    def _action_to_control(self, action_vector: np.ndarray) -> carla.VehicleControl:
        steer = float(action_vector[0])
        speed_kmh = compute_speed_kmh(self.vehicle) if self.vehicle is not None else 0.0
""",
        """    def _action_to_control(self, action_vector: np.ndarray) -> carla.VehicleControl:
        steer = float(action_vector[0])
        requested_steer = float(steer)
        speed_kmh = compute_speed_kmh(self.vehicle) if self.vehicle is not None else 0.0
        speed_error = 0.0
""",
    )

    text = replace_once(
        text,
        """            self.last_target_speed_kmh = float(target_speed_kmh)
            self.last_raw_longitudinal_action = float(raw_longitudinal)
        else:
            throttle = float(action_vector[1]) if action_vector.shape[0] > 1 else 0.0
            brake = float(action_vector[2]) if action_vector.shape[0] > 2 else 0.0
            self.last_target_speed_kmh = 0.0
            self.last_raw_longitudinal_action = 0.0

        if bool(self.action_smoothing.get("enabled", False)):
""",
        """            self.last_target_speed_kmh = float(target_speed_kmh)
            self.last_raw_longitudinal_action = float(raw_longitudinal)
        else:
            throttle = float(action_vector[1]) if action_vector.shape[0] > 1 else 0.0
            brake = float(action_vector[2]) if action_vector.shape[0] > 2 else 0.0
            self.last_target_speed_kmh = 0.0
            self.last_raw_longitudinal_action = 0.0

        requested_throttle = float(throttle)
        requested_brake = float(brake)
        self.last_speed_error_kmh = float(speed_error)
        self.last_raw_requested_throttle = float(requested_throttle)
        self.last_raw_requested_brake = float(requested_brake)

        if bool(self.action_smoothing.get("enabled", False)):
""",
    )

    text = replace_once(
        text,
        """        control = carla.VehicleControl(
            steer=steer,
            throttle=throttle,
            brake=brake,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
        )
""",
        """        self.last_smoothed_steer = float(steer)
        self.last_smoothed_throttle = float(throttle)
        self.last_smoothed_brake = float(brake)
        self.last_requested_applied_action_l1 = float(
            abs(requested_steer - steer)
            + abs(requested_throttle - throttle)
            + abs(requested_brake - brake)
        )

        control = carla.VehicleControl(
            steer=steer,
            throttle=throttle,
            brake=brake,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
        )
""",
    )

    text = replace_once(
        text,
        """        is_low_speed, stuck = self._update_stuck_status(speed_mps, route_metrics.reached_goal, current_step)
        offroad_candidate, offroad, offroad_threshold = self._update_offroad_status(
""",
        """        rgb_image = sensor_snapshot.get("rgb")
        rgb_brightness = 0.0
        rgb_contrast = 0.0
        rgb_dropout_flag = True
        if rgb_image is not None and hasattr(rgb_image, "raw_data"):
            try:
                raw_rgb = np.frombuffer(rgb_image.raw_data, dtype=np.uint8).reshape((int(rgb_image.height), int(rgb_image.width), 4))[:, :, :3]
                rgb_brightness = float(raw_rgb.mean() / 255.0)
                rgb_contrast = float(raw_rgb.std() / 255.0)
                rgb_dropout_flag = False
            except Exception:
                rgb_dropout_flag = True

        lidar_points = int(sensor_snapshot.get("lidar_points", 0))
        lidar_target_points = max(1.0, float(self.lidar_sensor.get("points_per_second", 20000)) * float(self.fixed_delta_seconds))
        lidar_valid_ratio = float(min(1.0, max(0.0, lidar_points / lidar_target_points)))
        road_mask = self._semantic_mask(sensor_snapshot.get("semantic"), labels={6})
        lane_mask = self._semantic_mask(sensor_snapshot.get("semantic"), labels={7})
        aux_available = sensor_snapshot.get("semantic") is not None

        is_low_speed, stuck = self._update_stuck_status(speed_mps, route_metrics.reached_goal, current_step)
        offroad_candidate, offroad, offroad_threshold = self._update_offroad_status(
""",
    )

    text = replace_once(
        text,
        """            "timestamp": timestamp_string(),
            "frame": frame,
            "step": current_step,
            "spawn_point_index": self.spawn_point_index,
            "speed_kmh": speed_kmh,
""",
        """            "timestamp": timestamp_string(),
            "frame": frame,
            "step": current_step,
            "episode_id": self.episode_id,
            "weather": self.weather,
            "traffic_mode": self.traffic_mode,
            "traffic_vehicle_count_applied": self.traffic_vehicle_count if self.traffic_mode != "none" else 0,
            "traffic_target_vehicle_count": self.traffic_vehicle_count,
            "traffic_spawn_failures": 0,
            "traffic_cleanup_destroy_failures": 0,
            "traffic_cleanup_residual_count": 0,
            "fixed_delta_seconds": self.fixed_delta_seconds,
            "spawn_point_index": self.spawn_point_index,
            "speed_kmh": speed_kmh,
""",
    )

    text = replace_once(
        text,
        """            "lidar_points": sensor_snapshot["lidar_points"],
            "action": action_vector.copy(),
            "target_speed_kmh": float(self.last_target_speed_kmh),
            "raw_longitudinal_action": float(self.last_raw_longitudinal_action),
            "action_smoothing_applied": bool(self.last_action_smoothing_applied),
            "maneuver_command": self._current_maneuver_command(),
""",
        """            "lidar_points": lidar_points,
            "lidar_point_count": lidar_points,
            "lidar_valid_ratio": lidar_valid_ratio,
            "lidar_sparsity": 1.0 - lidar_valid_ratio,
            "lidar_drop_ratio": 1.0 - lidar_valid_ratio,
            "rgb_brightness": rgb_brightness,
            "rgb_contrast": rgb_contrast,
            "rgb_entropy": 0.0,
            "rgb_blur_proxy": 0.0,
            "rgb_dropout_flag": rgb_dropout_flag,
            "action": action_vector.copy(),
            "action_contract_version": self.action_contract_version,
            "policy_action_space": "steer_target_speed" if action_vector.shape == (2,) else "steer_throttle_brake",
            "policy_action_steer": float(action_vector[0]) if action_vector.size > 0 else 0.0,
            "policy_action_longitudinal": float(action_vector[1]) if action_vector.size > 1 else 0.0,
            "requested_action_steer": float(action_vector[0]) if action_vector.size > 0 else 0.0,
            "requested_action_throttle": float(action_vector[1]) if action_vector.shape == (3,) else float(self.last_raw_requested_throttle),
            "requested_action_brake": float(action_vector[2]) if action_vector.shape == (3,) else float(self.last_raw_requested_brake),
            "longitudinal_control_mode": str(self.longitudinal_control.get("mode", "direct")),
            "target_speed_kmh": float(self.last_target_speed_kmh),
            "raw_longitudinal_action": float(self.last_raw_longitudinal_action),
            "speed_error_kmh": float(self.last_speed_error_kmh),
            "raw_requested_throttle": float(self.last_raw_requested_throttle),
            "raw_requested_brake": float(self.last_raw_requested_brake),
            "smoothed_steer": float(self.last_smoothed_steer),
            "smoothed_throttle": float(self.last_smoothed_throttle),
            "smoothed_brake": float(self.last_smoothed_brake),
            "action_smoothing_applied": bool(self.last_action_smoothing_applied),
            "action_smoothing_enabled": bool(self.action_smoothing.get("enabled", False)),
            "requested_applied_action_l1": float(self.last_requested_applied_action_l1),
            "speed_kmh_at_action_resolution": speed_kmh,
            "curve_proxy": abs(lookahead_turn_delta_rad),
            "suppress_conflicting_pedals_enabled": bool(self.suppress_conflicting_pedals),
            "pedal_conflict_threshold": 0.05,
            "pedal_conflict_resolution": "stronger_pedal_wins" if self.suppress_conflicting_pedals else "",
            "low_speed_brake_suppression_enabled": bool(self.low_speed_brake_suppression),
            "low_speed_brake_speed_threshold_kmh": float(self.low_speed_brake_speed_threshold_kmh),
            "low_speed_brake_max": float(self.low_speed_brake_max),
            "low_speed_brake_resolution": "cap_brake" if self.low_speed_brake_suppression else "",
            "low_speed_throttle_floor_enabled": False,
            "low_speed_throttle_floored": False,
            "lateral_recovery_enabled": bool(self.lateral_recovery.get("enabled", False)),
            "lateral_recovery_active": False,
            "lateral_recovery_lane_error": 0.0,
            "lateral_recovery_heading_error": 0.0,
            "lateral_recovery_correction": 0.0,
            "lateral_recovery_raw_steer": float(action_vector[0]) if action_vector.size > 0 else 0.0,
            "lateral_recovery_corrected_steer": float(control.steer),
            "lateral_recovery_curve_proxy": abs(lookahead_turn_delta_rad),
            "lateral_recovery_branch_active": bool(self.maneuver_command_enabled),
            "lateral_recovery_turn_boost_active": False,
            "lateral_recovery_effective_max_abs_correction": float(self.lateral_recovery.get("max_abs_correction", 0.0)),
            "lateral_recovery_lane_offset_threshold_m": float(self.lateral_recovery.get("lane_offset_threshold_m", 0.0)),
            "lateral_recovery_heading_error_threshold_deg": float(self.lateral_recovery.get("heading_error_threshold_deg", 0.0)),
            "lateral_recovery_lane_gain": float(self.lateral_recovery.get("lane_gain", 0.0)),
            "lateral_recovery_heading_gain": float(self.lateral_recovery.get("heading_gain", 0.0)),
            "lateral_recovery_max_abs_correction": float(self.lateral_recovery.get("max_abs_correction", 0.0)),
            "lateral_recovery_turn_boost_enabled": bool(self.lateral_recovery.get("turn_boost_enabled", False)),
            "weather_severity": float(self.corruption_config.get("weather_severity", 0.0)),
            "drop_rgb": float(self.corruption_config.get("drop_rgb", 0.0)),
            "drop_lidar": float(self.corruption_config.get("drop_lidar", 0.0)),
            "drop_ego": float(self.corruption_config.get("drop_ego", 0.0)),
            "drop_route": float(self.corruption_config.get("drop_route", 0.0)),
            "corruption_level": float(self.corruption_config.get("level", 0.0)),
            "corruption_window_id": int(self.corruption_config.get("window_id", 0)),
            "corruption_mode": str(self.corruption_config.get("mode", "")),
            "sensor_dropout_mask_all_zero": not any(
                float(self.corruption_config.get(key, 0.0)) > 0.0
                for key in ("drop_rgb", "drop_lidar", "drop_ego", "drop_route")
            ),
            "rgb_image_size_x": int(self.rgb_sensor.get("image_size_x", 800)),
            "rgb_image_size_y": int(self.rgb_sensor.get("image_size_y", 450)),
            "rgb_fov": float(self.rgb_sensor.get("fov", 100.0)),
            "lidar_channels": int(self.lidar_sensor.get("channels", 16)),
            "lidar_range": float(self.lidar_sensor.get("range", 40.0)),
            "lidar_points_per_second": int(self.lidar_sensor.get("points_per_second", 20000)),
            "action_smoothing_applied": bool(self.last_action_smoothing_applied),
            "maneuver_command": self._current_maneuver_command(),
""",
    )

    text = replace_once(
        text,
        """            "aux_road_mask_target": self._semantic_mask(sensor_snapshot.get("semantic"), labels={6}),
            "aux_lane_edge_target": self._semantic_mask(sensor_snapshot.get("semantic"), labels={7}),
            "aux_mask_targets_available": sensor_snapshot.get("semantic") is not None,
            "aux_mask_target_source": "semantic_camera" if sensor_snapshot.get("semantic") is not None else "",
""",
        """            "aux_road_mask_target": road_mask,
            "aux_lane_edge_target": lane_mask,
            "aux_mask_targets_available": aux_available,
            "aux_mask_target_source": "semantic_camera" if aux_available else "",
            "aux_road_mask_positive_ratio": float(np.asarray(road_mask, dtype=np.float32).mean()) if road_mask is not None else 0.0,
            "aux_lane_edge_positive_ratio": float(np.asarray(lane_mask, dtype=np.float32).mean()) if lane_mask is not None else 0.0,
""",
    )

    text = replace_once(
        text,
        """        transform = self.vehicle.get_transform()
        reward_breakdown = info.get("reward_breakdown", self._zero_reward_breakdown())
        row = {
""",
        """        transform = self.vehicle.get_transform()
        reward_breakdown = info.get("reward_breakdown", self._zero_reward_breakdown())
        diagnostics = dict(info.get("policy_diagnostics") or {})

        def _diag(name: str, default: float = 0.0) -> float:
            try:
                return float(diagnostics.get(name, default))
            except (TypeError, ValueError):
                return float(default)

        row = {
""",
    )

    text = replace_once(
        text,
        """            "step": info["step"],
            "x": round(transform.location.x, 3),
""",
        """            "step": info["step"],
            "episode_id": int(info.get("episode_id", 0)),
            "weather": info.get("weather", ""),
            "traffic_mode": info.get("traffic_mode", ""),
            "traffic_vehicle_count_applied": int(info.get("traffic_vehicle_count_applied", 0)),
            "traffic_target_vehicle_count": int(info.get("traffic_target_vehicle_count", 0)),
            "traffic_spawn_failures": int(info.get("traffic_spawn_failures", 0)),
            "traffic_cleanup_destroy_failures": int(info.get("traffic_cleanup_destroy_failures", 0)),
            "traffic_cleanup_residual_count": int(info.get("traffic_cleanup_residual_count", 0)),
            "fixed_delta_seconds": round(float(info.get("fixed_delta_seconds", self.fixed_delta_seconds)), 4),
            "x": round(transform.location.x, 3),
""",
    )

    text = replace_once(
        text,
        """            "route_progress_at_done": round(float(info.get("route_progress_at_done", 0.0)), 4),
            "reward": round(float(info.get("reward", 0.0)), 6),
""",
        """            "route_progress_at_done": round(float(info.get("route_progress_at_done", 0.0)), 4),
            "action_contract_version": info.get("action_contract_version", ""),
            "policy_action_space": info.get("policy_action_space", ""),
            "policy_action_steer": round(float(info.get("policy_action_steer", 0.0)), 6),
            "policy_action_longitudinal": round(float(info.get("policy_action_longitudinal", 0.0)), 6),
            "attention_mean_rgb": round(_diag("attention_mean_rgb", _diag("fusion_weight_rgb", 0.0)), 6),
            "attention_mean_lidar": round(_diag("attention_mean_lidar", _diag("fusion_weight_lidar", 0.0)), 6),
            "attention_mean_ego": round(_diag("attention_mean_ego", _diag("fusion_weight_ego", 0.0)), 6),
            "attention_mean_route": round(_diag("attention_mean_route", _diag("fusion_weight_route", 0.0)), 6),
            "fusion_weight_rgb": round(_diag("fusion_weight_rgb", 0.0), 6),
            "fusion_weight_lidar": round(_diag("fusion_weight_lidar", 0.0), 6),
            "fusion_weight_ego": round(_diag("fusion_weight_ego", 0.0), 6),
            "fusion_weight_route": round(_diag("fusion_weight_route", 0.0), 6),
            "fusion_entropy": round(_diag("fusion_entropy", 0.0), 6),
            "reliability_rgb": round(_diag("reliability_rgb", 1.0), 6),
            "reliability_lidar": round(_diag("reliability_lidar", 1.0), 6),
            "reliability_ego": round(_diag("reliability_ego", 1.0), 6),
            "reliability_route": round(_diag("reliability_route", 1.0), 6),
            "risk_scalar": round(_diag("risk_scalar", _diag("risk_scalar_mean", 0.0)), 6),
            "risk_context_l1_mean": round(_diag("risk_context_l1_mean", 0.0), 6),
            "attention_perception_sum": round(_diag("attention_perception_sum", _diag("fusion_weight_rgb", 0.0) + _diag("fusion_weight_lidar", 0.0)), 6),
            "attention_state_sum": round(_diag("attention_state_sum", _diag("fusion_weight_ego", 0.0) + _diag("fusion_weight_route", 0.0)), 6),
            "route_token_dropout_rate": round(_diag("route_token_dropout_rate", 0.0), 6),
            "requested_action_steer": round(float(info.get("requested_action_steer", 0.0)), 6),
            "requested_action_throttle": round(float(info.get("requested_action_throttle", 0.0)), 6),
            "requested_action_brake": round(float(info.get("requested_action_brake", 0.0)), 6),
            "longitudinal_control_mode": info.get("longitudinal_control_mode", ""),
            "target_speed_kmh": round(float(info.get("target_speed_kmh", 0.0)), 6),
            "speed_error_kmh": round(float(info.get("speed_error_kmh", 0.0)), 6),
            "raw_requested_throttle": round(float(info.get("raw_requested_throttle", 0.0)), 6),
            "raw_requested_brake": round(float(info.get("raw_requested_brake", 0.0)), 6),
            "smoothed_steer": round(float(info.get("smoothed_steer", 0.0)), 6),
            "smoothed_throttle": round(float(info.get("smoothed_throttle", 0.0)), 6),
            "smoothed_brake": round(float(info.get("smoothed_brake", 0.0)), 6),
            "lateral_recovery_enabled": bool(info.get("lateral_recovery_enabled", False)),
            "lateral_recovery_active": bool(info.get("lateral_recovery_active", False)),
            "lateral_recovery_lane_error": round(float(info.get("lateral_recovery_lane_error", 0.0)), 6),
            "lateral_recovery_heading_error": round(float(info.get("lateral_recovery_heading_error", 0.0)), 6),
            "lateral_recovery_correction": round(float(info.get("lateral_recovery_correction", 0.0)), 6),
            "lateral_recovery_raw_steer": round(float(info.get("lateral_recovery_raw_steer", 0.0)), 6),
            "lateral_recovery_corrected_steer": round(float(info.get("lateral_recovery_corrected_steer", 0.0)), 6),
            "lateral_recovery_curve_proxy": round(float(info.get("lateral_recovery_curve_proxy", 0.0)), 6),
            "lateral_recovery_branch_active": bool(info.get("lateral_recovery_branch_active", False)),
            "lateral_recovery_turn_boost_active": bool(info.get("lateral_recovery_turn_boost_active", False)),
            "lateral_recovery_effective_max_abs_correction": round(float(info.get("lateral_recovery_effective_max_abs_correction", 0.0)), 6),
            "lateral_recovery_lane_offset_threshold_m": round(float(info.get("lateral_recovery_lane_offset_threshold_m", 0.0)), 6),
            "lateral_recovery_heading_error_threshold_deg": round(float(info.get("lateral_recovery_heading_error_threshold_deg", 0.0)), 6),
            "lateral_recovery_lane_gain": round(float(info.get("lateral_recovery_lane_gain", 0.0)), 6),
            "lateral_recovery_heading_gain": round(float(info.get("lateral_recovery_heading_gain", 0.0)), 6),
            "lateral_recovery_max_abs_correction": round(float(info.get("lateral_recovery_max_abs_correction", 0.0)), 6),
            "lateral_recovery_turn_boost_enabled": bool(info.get("lateral_recovery_turn_boost_enabled", False)),
            "action_smoothing_enabled": bool(info.get("action_smoothing_enabled", False)),
            "requested_applied_action_l1": round(float(info.get("requested_applied_action_l1", 0.0)), 6),
            "suppress_conflicting_pedals_enabled": bool(info.get("suppress_conflicting_pedals_enabled", False)),
            "pedal_conflict_threshold": round(float(info.get("pedal_conflict_threshold", 0.0)), 6),
            "pedal_conflict_resolution": info.get("pedal_conflict_resolution", ""),
            "low_speed_brake_suppression_enabled": bool(info.get("low_speed_brake_suppression_enabled", False)),
            "low_speed_brake_speed_threshold_kmh": round(float(info.get("low_speed_brake_speed_threshold_kmh", 0.0)), 6),
            "low_speed_brake_max": round(float(info.get("low_speed_brake_max", 0.0)), 6),
            "low_speed_brake_resolution": info.get("low_speed_brake_resolution", ""),
            "low_speed_throttle_floor_enabled": bool(info.get("low_speed_throttle_floor_enabled", False)),
            "speed_kmh_at_action_resolution": round(float(info.get("speed_kmh_at_action_resolution", 0.0)), 6),
            "curve_proxy": round(float(info.get("curve_proxy", 0.0)), 6),
            "weather_severity": round(float(info.get("weather_severity", 0.0)), 6),
            "drop_rgb": round(float(info.get("drop_rgb", 0.0)), 6),
            "drop_lidar": round(float(info.get("drop_lidar", 0.0)), 6),
            "drop_ego": round(float(info.get("drop_ego", 0.0)), 6),
            "drop_route": round(float(info.get("drop_route", 0.0)), 6),
            "corruption_level": round(float(info.get("corruption_level", 0.0)), 6),
            "corruption_window_id": int(info.get("corruption_window_id", 0)),
            "corruption_mode": info.get("corruption_mode", ""),
            "sensor_dropout_mask_all_zero": bool(info.get("sensor_dropout_mask_all_zero", True)),
            "reward": round(float(info.get("reward", 0.0)), 6),
""",
    )

    text = replace_once(
        text,
        """            "lane_markings": "|".join(info["lane_markings"]),
            "lidar_points": info["lidar_points"],
            "spawn_point_index": int(info.get("spawn_point_index", -1)),
            "rgb_image_path": image_path,
        }
""",
        """            "lane_markings": "|".join(info["lane_markings"]),
            "lidar_points": info["lidar_points"],
            "lidar_point_count": int(info.get("lidar_point_count", info.get("lidar_points", 0))),
            "lidar_valid_ratio": round(float(info.get("lidar_valid_ratio", 0.0)), 6),
            "lidar_sparsity": round(float(info.get("lidar_sparsity", 0.0)), 6),
            "lidar_drop_ratio": round(float(info.get("lidar_drop_ratio", 0.0)), 6),
            "rgb_brightness": round(float(info.get("rgb_brightness", 0.0)), 6),
            "rgb_contrast": round(float(info.get("rgb_contrast", 0.0)), 6),
            "rgb_entropy": round(float(info.get("rgb_entropy", 0.0)), 6),
            "rgb_blur_proxy": round(float(info.get("rgb_blur_proxy", 0.0)), 6),
            "rgb_dropout_flag": bool(info.get("rgb_dropout_flag", False)),
            "aux_mask_targets_available": bool(info.get("aux_mask_targets_available", False)),
            "aux_mask_target_source": info.get("aux_mask_target_source", ""),
            "aux_road_mask_positive_ratio": round(float(info.get("aux_road_mask_positive_ratio", 0.0)), 6),
            "aux_lane_edge_positive_ratio": round(float(info.get("aux_lane_edge_positive_ratio", 0.0)), 6),
            "spawn_point_index": int(info.get("spawn_point_index", -1)),
            "rgb_image_size_x": int(info.get("rgb_image_size_x", 0)),
            "rgb_image_size_y": int(info.get("rgb_image_size_y", 0)),
            "rgb_fov": round(float(info.get("rgb_fov", 0.0)), 6),
            "lidar_channels": int(info.get("lidar_channels", 0)),
            "lidar_range": round(float(info.get("lidar_range", 0.0)), 6),
            "lidar_points_per_second": int(info.get("lidar_points_per_second", 0)),
            "rgb_image_path": image_path,
        }
""",
    )

    text = replace_once(
        text,
        """        self._close_log_file()
        self.csv_file = open(self.logs_path, "w", newline="", encoding="utf-8")
""",
        """        self._close_log_file()
        log_exists = os.path.exists(self.logs_path) and os.path.getsize(self.logs_path) > 0
        self.csv_file = open(self.logs_path, "a" if log_exists else "w", newline="", encoding="utf-8")
""",
    )

    text = replace_once(
        text,
        """        self.csv_writer.writeheader()
""",
        """        if not log_exists:
            self.csv_writer.writeheader()
""",
    )

    TARGET.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
