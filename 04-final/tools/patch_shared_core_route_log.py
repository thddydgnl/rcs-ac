from __future__ import annotations

from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"{label} not found")
    return text.replace(old, new, 1)


def main() -> None:
    path = Path("env/carla_env.py")
    backup = Path("env/carla_env.py.route_schema_backup_20260504")
    text = path.read_text(encoding="utf-8")
    backup.write_text(text, encoding="utf-8")

    text = replace_once(
        text,
        "    from shared_core.env.route_manager import RouteManager\n    from shared_core.env.sensors import (",
        "    from shared_core.env.route_manager import RouteManager\n"
        "    from shared_core.env.log_schema import ENV_STEP_LOG_FIELDS\n"
        "    from shared_core.env.sensors import (",
        "shared import",
    )
    text = replace_once(
        text,
        "    from env.reward import RewardCalculator, RewardConfig\n"
        "    from env.route_manager import RouteManager\n"
        "    from env.sensors import CollisionSensor, LidarSensor, LaneInvasionSensor, RgbCamera, SensorSyncError",
        "    from env.reward import RewardCalculator, RewardConfig\n"
        "    from env.route_manager import RouteManager\n"
        "    try:\n"
        "        from env.log_schema import ENV_STEP_LOG_FIELDS\n"
        "    except ModuleNotFoundError:\n"
        "        ENV_STEP_LOG_FIELDS = ()\n"
        "    from env.sensors import CollisionSensor, LidarSensor, LaneInvasionSensor, RgbCamera, SensorSyncError",
        "fallback import",
    )
    text = replace_once(
        text,
        '        self.logs_path = os.path.join(self.output_dir, "logs.csv")',
        '        self.logs_path = os.path.join(self.output_dir, "env_step_log.csv")',
        "log path",
    )

    old_open = """        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                "timestamp",
                "frame",
                "step",
                "x",
                "y",
                "z",
                "yaw",
                "speed",
                "steer",
                "throttle",
                "brake",
                "target_waypoint_index",
                "route_progress",
                "heading_error",
                "lane_offset",
                "target_distance",
                "final_target_distance",
                "min_target_distance",
                "collision",
                "lane_invasion",
                "offroad",
                "offroad_candidate",
                "offroad_steps",
                "offroad_threshold",
                "offroad_grace_steps",
                "offroad_grace_active",
                "first_offroad_step",
                "stuck_trigger_step",
                "stuck",
                "low_speed_steps",
                "recent_steer_mean",
                "recent_throttle_mean",
                "recent_brake_mean",
                "recent_control_window",
                "recent_speed_mean_kmh",
                "pedal_overlap_suppressed",
                "low_speed_brake_suppressed",
                "reached_goal",
                "goal_check_hit",
                "goal_radius",
                "route_progress_at_done",
                "reward",
                "progress_reward",
                "dense_progress_reward",
                "lane_reward",
                "lane_offset_penalty",
                "heading_error_penalty",
                "heading_recovery_reward",
                "speed_reward",
                "speed_alignment_factor",
                "speed_alignment_bonus",
                "curve_overspeed_penalty",
                "collision_penalty",
                "offroad_penalty",
                "lane_invasion_penalty",
                "control_cost",
                "high_speed_steer_penalty",
                "goal_bonus",
                "stuck_penalty",
                "done",
                "done_reason",
                "collision_actor",
                "collision_intensity",
                "lane_markings",
                "lidar_points",
                "spawn_point_index",
                "rgb_image_path",
            ],
        )"""
    new_open = """        fieldnames = list(ENV_STEP_LOG_FIELDS) if ENV_STEP_LOG_FIELDS else [
            "timestamp",
            "frame",
            "step",
            "x",
            "y",
            "z",
            "yaw",
            "speed",
            "steer",
            "throttle",
            "brake",
            "target_waypoint_index",
            "route_progress",
            "heading_error",
            "lane_offset",
            "target_distance",
            "final_target_distance",
            "min_target_distance",
            "collision",
            "lane_invasion",
            "offroad",
            "done",
            "done_reason",
            "spawn_point_index",
            "rgb_image_path",
        ]
        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )"""
    text = replace_once(text, old_open, new_open, "open log block")

    text = replace_once(
        text,
        "        recent_control_summary = self._recent_control_summary()\n"
        "        is_low_speed, stuck = self._update_stuck_status(speed_mps, route_metrics.reached_goal, current_step)",
        "        recent_control_summary = self._recent_control_summary()\n"
        '        route_vector = np.asarray(observation.get("route", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)\n'
        "        route_metadata = (\n"
        "            self.route_manager.get_route_metadata()\n"
        '            if hasattr(self.route_manager, "get_route_metadata")\n'
        "            else {}\n"
        "        )\n"
        '        route_representation = getattr(self.state_builder.config, "route_representation", "")\n'
        "        target_x_local = float(route_vector[0]) if route_vector.size > 0 else 0.0\n"
        "        target_y_local = float(route_vector[1]) if route_vector.size > 1 else 0.0\n"
        "        lookahead_turn_delta_rad = float(route_vector[7]) if route_vector.size > 7 else 0.0\n"
        "        is_low_speed, stuck = self._update_stuck_status(speed_mps, route_metrics.reached_goal, current_step)",
        "build_info route variables",
    )

    text = replace_once(
        text,
        '            "target_waypoint_index": route_metrics.target_index,\n'
        '            "target_distance": target_distance,',
        '            "target_waypoint_index": route_metrics.target_index,\n'
        '            "route_representation": route_representation,\n'
        '            "target_x_local": target_x_local,\n'
        '            "target_y_local": target_y_local,\n'
        '            "lookahead_turn_delta_rad": lookahead_turn_delta_rad,\n'
        '            "route_waypoint_count": int(route_metadata.get("route_waypoint_count", 0)),\n'
        '            "route_total_length_m": float(route_metadata.get("route_total_length_m", 0.0)),\n'
        '            "route_lane_switch_count": int(route_metadata.get("route_lane_switch_count", 0)),\n'
        '            "route_road_switch_count": int(route_metadata.get("route_road_switch_count", 0)),\n'
        '            "route_heading_jump_max_deg": float(route_metadata.get("route_heading_jump_max_deg", 0.0)),\n'
        '            "route_continuity_mean_abs_error_m": float(route_metadata.get("route_continuity_mean_abs_error_m", 0.0)),\n'
        '            "route_continuity": float(route_metadata.get("route_continuity_score", 1.0)),\n'
        '            "route_heading_mismatch": abs(float(route_metrics.heading_error)),\n'
        '            "route_availability_flag": 1.0 if int(route_metadata.get("route_waypoint_count", 0)) > 0 else 0.0,\n'
        '            "target_distance": target_distance,',
        "build_info return route fields",
    )

    text = replace_once(
        text,
        '            "target_waypoint_index": info["target_waypoint_index"],\n'
        '            "route_progress": round(info["route_progress"], 4),',
        '            "target_waypoint_index": info["target_waypoint_index"],\n'
        '            "route_representation": info.get("route_representation", ""),\n'
        '            "target_x_local": round(float(info.get("target_x_local", 0.0)), 6),\n'
        '            "target_y_local": round(float(info.get("target_y_local", 0.0)), 6),\n'
        '            "lookahead_turn_delta_rad": round(float(info.get("lookahead_turn_delta_rad", 0.0)), 6),\n'
        '            "route_progress": round(info["route_progress"], 4),\n'
        '            "route_waypoint_count": int(info.get("route_waypoint_count", 0)),\n'
        '            "route_total_length_m": round(float(info.get("route_total_length_m", 0.0)), 4),\n'
        '            "route_lane_switch_count": int(info.get("route_lane_switch_count", 0)),\n'
        '            "route_road_switch_count": int(info.get("route_road_switch_count", 0)),\n'
        '            "route_heading_jump_max_deg": round(float(info.get("route_heading_jump_max_deg", 0.0)), 4),\n'
        '            "route_continuity_mean_abs_error_m": round(float(info.get("route_continuity_mean_abs_error_m", 0.0)), 6),\n'
        '            "route_continuity": round(float(info.get("route_continuity", 1.0)), 6),\n'
        '            "route_heading_mismatch": round(float(info.get("route_heading_mismatch", 0.0)), 4),\n'
        '            "route_availability_flag": round(float(info.get("route_availability_flag", 0.0)), 4),',
        "write row route fields",
    )

    text = replace_once(
        text,
        "        self.csv_writer.writerow(row)\n        self.csv_file.flush()",
        '        for field in self.csv_writer.fieldnames or []:\n'
        '            row.setdefault(field, "")\n'
        "        self.csv_writer.writerow(row)\n"
        "        self.csv_file.flush()",
        "writerow fill",
    )

    path.write_text(text, encoding="utf-8")
    print("PATCHED_SHARED_CORE_CARLA_ENV")


if __name__ == "__main__":
    main()
