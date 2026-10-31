import argparse
import os
import sys
from pathlib import Path


def _bootstrap_carla():
    try:
        import carla  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    search_roots = []

    env_root = os.environ.get("CARLA_ROOT")
    if env_root:
        search_roots.append(Path(env_root))

    search_roots.extend(
        [
            Path.home() / "Downloads" / "CARLA_0.9.16",
            Path("C:/CARLA_0.9.16"),
        ]
    )

    for root in search_roots:
        dist_dir = root / "PythonAPI" / "carla" / "dist"
        if not dist_dir.exists():
            continue

        matching = sorted(dist_dir.glob(f"carla-*-{py_tag}-{py_tag}-win_amd64.whl"))
        if matching:
            sys.path.append(str(matching[-1]))
            import carla  # noqa: F401
            return

        available = sorted(dist_dir.glob("carla-*.whl"))
        if available:
            versions = ", ".join(item.name for item in available)
            raise RuntimeError(
                "Found CARLA Python API, but it does not match this Python version. "
                f"Current Python is {sys.version.split()[0]} and available wheels are: {versions}. "
                "Run this project with a matching Python, for example `py -3.12 main.py ...`."
            )

    raise RuntimeError(
        "CARLA Python API was not found. Install the CARLA wheel for this Python version "
        "or set the CARLA_ROOT environment variable to your CARLA installation directory."
    )


_bootstrap_carla()

from env.carla_env import CarlaEnv


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Minimal CARLA route-following runner.")
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--town", default="Town04", help="Town map to load")
    parser.add_argument("--steps", type=int, default=300, help="Maximum simulation steps")
    parser.add_argument("--route-length", type=int, default=40, help="Number of waypoints in route")
    parser.add_argument("--route-spacing", type=float, default=3.0, help="Meters between route waypoints")
    parser.add_argument("--save-dir", default=os.path.join("outputs", "carla_run"), help="Output directory")
    parser.add_argument("--image-every", type=int, default=10, help="Save one RGB frame every N steps")
    parser.add_argument("--timeout", type=float, default=10.0, help="CARLA client timeout seconds")
    return parser


def main():
    args = build_arg_parser().parse_args()
    env = CarlaEnv(
        host=args.host,
        port=args.port,
        town=args.town,
        output_dir=args.save_dir,
        image_every=args.image_every,
        client_timeout=args.timeout,
        route_length=args.route_length,
        route_spacing=args.route_spacing,
    )

    try:
        reset_info = env.reset()
        print(
            f"[reset] frame={reset_info['frame']} "
            f"route_waypoints={reset_info['route_waypoints']} "
            f"done={reset_info['done']} "
            f"done_reason={reset_info['info']['done_reason']}"
        )

        for step_idx in range(args.steps):
            step_info = env.step()
            print(
                "[step] "
                f"idx={step_idx:04d} "
                f"frame={step_info['frame']} "
                f"speed_kmh={step_info['speed_kmh']:.2f} "
                f"target_idx={step_info['target_waypoint_index']} "
                f"route_done={step_info['route_completed']} "
                f"reward={step_info['reward']:.3f} "
                f"done={step_info['done']} "
                f"done_reason={step_info['done_reason']} "
                f"collision={step_info['collision']} "
                f"lane_invasion={step_info['lane_invasion']}"
            )

            if step_info["done"]:
                print(f"[done] Episode finished with reason={step_info['done_reason']}.")
                break

    except KeyboardInterrupt:
        print("[stop] Interrupted by user.")
    finally:
        env.close()
        print(f"[cleanup] Outputs written to: {os.path.abspath(args.save_dir)}")


if __name__ == "__main__":
    main()
