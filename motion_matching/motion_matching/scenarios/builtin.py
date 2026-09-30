from typing import Any

import numpy as np
import random

VEL_CMD_MAPPING = {
    0: "stand",
    1: "forward",
    2: "left_45",
    3: "left_90",
    4: "right_45",
    5: "right_90",
    6: "run_forward",
    7: "run_left_45",
    8: "run_left_90",
    9: "run_right_45",
    10: "run_right_90",
    11: "backward",
}

VEL_CMD_MAPPING_REVERSE = {v: k for k, v in VEL_CMD_MAPPING.items()}

VEL_CMD_DIM = 15

WALKING_SPEED = 0.8
WALKING_SPEEDUP = 1.0
RUNNING_SPEED = 2.2
RUNNING_SPEEDUP = 1.0


def random_locomotion(
    walking_speed: float = WALKING_SPEED,
    running_speed: float = RUNNING_SPEED,
    num_skills: int = 10,
    num_configs: int = 100,
) -> tuple[list[dict[str, Any]], float]:
    """Generate stand-to-stand locomotion scenarios with no internal stops.

    The body has a geometrically distributed length capped by ``num_skills``.
    With the default stop probability, the capped distribution has an expected
    length of about 5.35 skills.
    """
    scenarios_list = [
        ("forward", 0.0),
        ("left_45", 45.0),
        ("left_90", 90.0),
        ("right_45", -45.0),
        ("right_90", -90.0),
        ("run_forward", 0.0),
        ("run_left_45", 45.0),
        ("run_left_90", 90.0),
        ("run_right_45", -45.0),
        ("run_right_90", -90.0),
    ]
    np.random.seed(42)
    random.seed(42)

    stand_duration = 2.0
    stop_probability = 0.15

    def stand_step() -> dict[str, Any]:
        return {
            "type": "locomotion",
            "speed": 0.0,
            "angle": 0.0,
            "duration": stand_duration,
            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["stand"],
        }

    def locomotion_step(skill: tuple[str, float]) -> tuple[dict[str, Any], str]:
        duration = float(np.random.uniform(0.5, 3.0))
        speed = walking_speed
        if skill[0].startswith("run_"):
            duration = max(duration, 1.2)
            speed = running_speed
        step = {
            "type": "locomotion",
            "speed": speed,
            "angle": skill[1] * duration,
            "duration": duration,
            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE[skill[0]],
        }
        return step, f"{skill[0]}_{duration:.1f}s_"

    configs = []
    for i in range(num_configs):
        steps = [stand_step()]
        name = f"cfg{i:04d}_stand_{stand_duration:.1f}s_"

        for _ in range(num_skills):
            step, label = locomotion_step(random.choice(scenarios_list))
            steps.append(step)
            name += label
            if np.random.random() < stop_probability:
                break

        steps.append(stand_step())
        name += f"stand_{stand_duration:.1f}s_"

        config = {"name": name[:-1], "steps": steps}
        # Alternate obstacle-free and cluttered scenes so the default batch is
        # split evenly without correlating terrain choice with one contiguous
        # portion of the generated motions.
        if i % 2 == 1:
            # Place deterministic, nearby clutter after the root trajectory
            # has been generated. The boxes are training terrain only; they
            # do not affect motion matching itself.
            config["offpath_obstacles"] = {
                "seed": 42000 + i,
                "min_count": 5,
                "max_count": 15,
                "path_clearance": 0.35,
                "min_center_distance": 0.3,
                "max_path_distance": 1.5,
                "obstacle_gap": 0.05,
            }
        configs.append(config)
    return configs, WALKING_SPEEDUP


def low_speed_step(walking_speed: float = WALKING_SPEED) -> tuple[list[dict[str, Any]], float]:
    scenarios_list = [("forward", 0.0), ("left_45", 45.0), ("left_90", 90.0), ("right_45", -45.0), ("right_90", -90.0)]
    walk_duration_list = [0.5, 0.8, 1.2, 1.5]
    step_duration_list = [0, 0.5, 1.0]

    stand_duration = 1.0

    configs = []
    for name, angular_velocity in scenarios_list:
        for walk_duration in walk_duration_list:
            walk_duration = round(walk_duration, 1)
            angle = angular_velocity * walk_duration
            for step_duration in step_duration_list:
                if step_duration > 0:
                    steps = [
                        # Initial Stand
                        {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": stand_duration, "vel_cmd_idx": 0},
                        # Move forward
                        {"type": "locomotion", "speed": walking_speed, "angle": 0.0, "duration": 1, "vel_cmd_idx": 1},
                        # Continue moving while turning
                        {
                            "type": "locomotion",
                            "speed": walking_speed,
                            "angle": angle,
                            "duration": walk_duration,
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE[name],
                        },
                        # Trigger stepping skill (Duration and velocity are handled automatically)
                        {"type": "skill", "skill_name": "step_up", "vel_cmd_idx": 1},
                        # Walking on the stepping platform
                        {
                            "type": "locomotion",
                            "speed": walking_speed,
                            "angle": 0.0,
                            "duration": step_duration,
                            "vel_cmd_idx": 1,
                        },
                        # Descend from the stepping platform
                        {"type": "skill", "skill_name": "step_down", "vel_cmd_idx": 1},
                        # Continue moving
                        {"type": "locomotion", "speed": walking_speed, "angle": 0.0, "duration": np.random.uniform(1.0, 1.5), "vel_cmd_idx": 1},
                        # Final Stand
                        {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": 2.5, "vel_cmd_idx": 0},
                    ]
                else:
                    if angular_velocity == 90.0:
                        continue
                    steps = [
                        # Initial Stand
                        {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": stand_duration, "vel_cmd_idx": 0},
                        # Moving while turning
                        {
                            "type": "locomotion",
                            "speed": walking_speed,
                            "angle": angle,
                            "duration": walk_duration,
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE[name],
                        },
                        # Trigger stepping skill (Duration and velocity are handled automatically)
                        {"type": "skill", "skill_name": "step", "vel_cmd_idx": 1},
                        # Continue moving
                        {"type": "locomotion", "speed": walking_speed, "angle": 0.0, "duration": np.random.uniform(1.0, 1.5), "vel_cmd_idx": 1},
                        # Final Stand
                        {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": 2.5, "vel_cmd_idx": 0},
                    ]
                configs.append({"name": f"{name}_walk{walk_duration}s_step{step_duration}s", "steps": steps})
    return configs, WALKING_SPEEDUP


def high_speed_step(running_speed: float = RUNNING_SPEED) -> tuple[list[dict[str, Any]], float]:
    scenarios_list = [
        ("run_forward", 0.0),
        ("run_left_45", 45.0),
        ("run_left_90", 90.0),
        ("run_right_45", -45.0),
        ("run_right_90", -90.0),
    ]
    walk_duration_list = [0.6, 0.9, 1.2, 1.5]
    step_duration_list = [0, 0.5, 1.0]

    stand_duration = 1.0

    configs = []
    for name, angular_velocity in scenarios_list:
        for walk_duration in walk_duration_list:
            walk_duration = round(walk_duration, 1)
            angle = angular_velocity * walk_duration
            for step_duration in step_duration_list:
                if step_duration > 0:
                    steps = [
                        {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": stand_duration, "vel_cmd_idx": 0},
                        {
                            "type": "locomotion",
                            "speed": running_speed,
                            "angle": 0.0,
                            "duration": 1,
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                        },
                        {
                            "type": "locomotion",
                            "speed": running_speed,
                            "angle": angle,
                            "duration": walk_duration,
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE[name],
                        },
                        {
                            "type": "skill",
                            "skill_name": "step_high_speed_up",
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                        },
                        {
                            "type": "locomotion",
                            "speed": running_speed + 0.3,
                            "angle": 0.0,
                            "duration": 0.5,
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                        },
                        {
                            "type": "locomotion",
                            "speed": running_speed,
                            "angle": 0.0,
                            "duration": step_duration,
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                        },
                        {
                            "type": "skill",
                            "skill_name": "step_high_speed_down",
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                        },
                        {
                            "type": "locomotion",
                            "speed": running_speed,
                            "angle": 0.0,
                            "duration": np.random.uniform(1.0, 1.5),
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                        },
                        {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": 2.5, "vel_cmd_idx": 0},
                    ]
                else:
                    steps = [
                        {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": stand_duration, "vel_cmd_idx": 0},
                        {
                            "type": "locomotion",
                            "speed": running_speed,
                            "angle": 0.0,
                            "duration": 1,
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                        },
                        {
                            "type": "locomotion",
                            "speed": running_speed,
                            "angle": angle,
                            "duration": walk_duration,
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE[name],
                        },
                        {
                            "type": "skill",
                            "skill_name": "step_high_speed",
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                        },
                        {
                            "type": "locomotion",
                            "speed": running_speed,
                            "angle": 0.0,
                            "duration": np.random.uniform(1.0, 1.5),
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                        },
                        {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": 2.5, "vel_cmd_idx": 0},
                    ]
                configs.append({"name": f"{name}_{walk_duration}s_step{step_duration}s", "steps": steps})
    return configs, RUNNING_SPEEDUP


def low_speed_climb_58(walking_speed: float = WALKING_SPEED) -> tuple[list[dict[str, Any]], float]:
    scenarios_list = [("forward", 0.0), ("left_45", 45.0), ("left_90", 90.0), ("right_45", -45.0), ("right_90", -90.0)]

    stand_duration = 1.0
    walk_duration_list = [0.5, 1.0, 1.5, 2.0]
    platform_duration_list = [0, 1.2]

    configs = []
    for name, angular_velocity in scenarios_list:
        for walk_duration in walk_duration_list:
            walk_duration = round(walk_duration, 1)
            angle = angular_velocity * walk_duration
            for platform_duration in platform_duration_list:
                steps = [
                    {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": stand_duration, "vel_cmd_idx": 0},
                    {"type": "locomotion", "speed": walking_speed, "angle": 0.0, "duration": 1, "vel_cmd_idx": 1},
                    {
                        "type": "locomotion",
                        "speed": walking_speed,
                        "angle": angle,
                        "duration": walk_duration,
                        "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE[name],
                    },
                    {"type": "skill", "skill_name": "climb_58_up", "vel_cmd_idx": 1},
                ]

                if platform_duration > 0:
                    steps.append(
                        {
                            "type": "locomotion",
                            "speed": walking_speed,
                            "angle": 0.0,
                            "duration": platform_duration,
                            "vel_cmd_idx": 1,
                        }
                    )
                steps.extend(
                    [
                        {"type": "skill", "skill_name": "climb_58_down", "vel_cmd_idx": 1},
                        {"type": "locomotion", "speed": walking_speed, "angle": 0.0, "duration": np.random.uniform(1.0, 1.5), "vel_cmd_idx": 1},
                        {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": 2.5, "vel_cmd_idx": 0},
                    ]
                )
                configs.append({"name": f"{name}_walk{walk_duration}s_plat{platform_duration}s", "steps": steps})
    return configs, WALKING_SPEEDUP


def high_speed_climb_58(running_speed: float = RUNNING_SPEED) -> tuple[list[dict[str, Any]], float]:
    scenarios_list = [
        ("run_forward", 0.0),
        ("run_left_45", 45.0),
        ("run_left_90", 90.0),
        ("run_right_45", -45.0),
        ("run_right_90", -90.0),
    ]
    walk_duration_list = [0.6, 0.9, 1.2, 1.5]
    step_duration_list = [0.1, 1.0]

    stand_duration = 1.0

    configs = []
    for name, angular_velocity in scenarios_list:
        for walk_duration in walk_duration_list:
            walk_duration = round(walk_duration, 1)
            angle = angular_velocity * walk_duration
            for step_duration in step_duration_list:
                if step_duration > 0:
                    steps = [
                        {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": stand_duration, "vel_cmd_idx": 0},
                        {
                            "type": "locomotion",
                            "speed": running_speed,
                            "angle": 0.0,
                            "duration": 1,
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                        },
                        {
                            "type": "locomotion",
                            "speed": running_speed,
                            "angle": angle,
                            "duration": walk_duration,
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE[name],
                        },
                        {
                            "type": "skill",
                            "skill_name": "climb_58_up",
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                        },
                        {
                            "type": "locomotion",
                            "speed": running_speed,
                            "angle": 0.0,
                            "duration": step_duration,
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                        },
                        {
                            "type": "skill",
                            "skill_name": "climb_58_down",
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                        },
                        {
                            "type": "locomotion",
                            "speed": running_speed,
                            "angle": 0.0,
                            "duration": np.random.uniform(1.0, 1.5),
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                        },
                        {
                            "type": "segment",
                            "speed": 0.0,
                            "angle": 0.0,
                            "duration": 2.5,
                            "vel_cmd_idx": 0,
                        },
                    ]
                configs.append({"name": f"{name}_{walk_duration}s_step{step_duration}s", "steps": steps})
    return configs, RUNNING_SPEEDUP


def low_speed_climb_76(walking_speed: float = WALKING_SPEED) -> tuple[list[dict[str, Any]], float]:
    scenarios_list = [("forward", 0.0), ("left_45", 45.0), ("left_90", 90.0), ("right_45", -45.0), ("right_90", -90.0)]
    walk_duration_list = [0.6, 0.9, 1.2, 1.5]
    step_duration_list = [0.7, 1.2]

    stand_duration = 1.0

    configs = []
    for name, angular_velocity in scenarios_list:
        for walk_duration in walk_duration_list:
            walk_duration = round(walk_duration, 1)
            angle = angular_velocity * walk_duration
            for step_duration in step_duration_list:
                if step_duration > 0:
                    steps = [
                        {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": stand_duration, "vel_cmd_idx": 0},
                        {"type": "locomotion", "speed": walking_speed, "angle": 0.0, "duration": 1, "vel_cmd_idx": 1},
                        {
                            "type": "locomotion",
                            "speed": walking_speed,
                            "angle": angle,
                            "duration": walk_duration,
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE[name],
                        },
                        {
                            "type": "locomotion",
                            "speed": 0,
                            "angle": 0,
                            "duration": 0.4,
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE[name],
                        },
                        {"type": "skill", "skill_name": "climb_76_up", "vel_cmd_idx": 1},
                        {
                            "type": "locomotion",
                            "speed": walking_speed,
                            "angle": 0.0,
                            "duration": step_duration,
                            "vel_cmd_idx": 1,
                        },
                        {"type": "skill", "skill_name": "climb_76_down", "vel_cmd_idx": 1},
                        {"type": "locomotion", "speed": walking_speed, "angle": 0.0, "duration": np.random.uniform(1.0, 3.0), "vel_cmd_idx": 1},
                        {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": 2.5, "vel_cmd_idx": 0},
                    ]
                configs.append({"name": f"{name}_walk{walk_duration}s_step{step_duration}s", "steps": steps})
    return configs, WALKING_SPEEDUP


def high_speed_climb_76(running_speed: float = RUNNING_SPEED) -> tuple[list[dict[str, Any]], float]:
    scenarios_list = [
        ("run_forward", 0.0),
        ("run_left_45", 46.0),
        ("run_left_90", 90.0),
        ("run_right_45", -46.0),
        ("run_right_90", -90.0),
    ]
    walk_duration_list = [0.6, 1.0, 1.2, 1.5]
    step_duration_list = [0, 0.8]

    stand_duration = 1.0

    configs = []
    for name, angular_velocity in scenarios_list:
        for walk_duration in walk_duration_list:
            walk_duration = round(walk_duration, 1)
            angle = angular_velocity * walk_duration
            for step_duration in step_duration_list:
                if step_duration > 0:
                    steps = [
                        {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": stand_duration, "vel_cmd_idx": 0},
                        {
                            "type": "locomotion",
                            "speed": running_speed,
                            "angle": 0.0,
                            "duration": 1,
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                        },
                        {
                            "type": "locomotion",
                            "speed": running_speed,
                            "angle": angle,
                            "duration": walk_duration,
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE[name],
                        },
                        {
                            "type": "skill",
                            "skill_name": "climb_76_high_speed_up",
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                        },
                        {
                            "type": "locomotion",
                            "speed": 1.5,
                            "angle": 0.0,
                            "duration": 0.3,
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                        },
                        {
                            "type": "locomotion",
                            "speed": running_speed,
                            "angle": 0.0,
                            "duration": step_duration,
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                        },
                        {
                            "type": "locomotion",
                            "speed": 0.5,
                            "angle": 0.0,
                            "duration": 0.8,
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                        },
                        {
                            "type": "locomotion",
                            "speed": 0.2,
                            "angle": 0.0,
                            "duration": 0.5,
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                        },
                        {
                            "type": "skill",
                            "skill_name": "climb_76_high_speed_down",
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                        },
                        {
                            "type": "locomotion",
                            "speed": running_speed,
                            "angle": 0.0,
                            "duration": 1.0,
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                        },
                        {
                            "type": "segment",
                            "speed": 0.0,
                            "angle": 0.0,
                            "duration": 2.5,
                            "vel_cmd_idx": 0,
                        },
                    ]
                else:
                    steps = [
                        {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": stand_duration, "vel_cmd_idx": 0},
                        {
                            "type": "locomotion",
                            "speed": running_speed,
                            "angle": 0.0,
                            "duration": 1,
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                        },
                        {
                            "type": "locomotion",
                            "speed": running_speed,
                            "angle": angle,
                            "duration": walk_duration,
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE[name],
                        },
                        {
                            "type": "skill",
                            "skill_name": "climb_76_high_speed_up",
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                        },
                        {
                            "type": "locomotion",
                            "speed": 1.5,
                            "angle": 0.0,
                            "duration": 0.3,
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                        },
                        {
                            "type": "locomotion",
                            "speed": 0.2,
                            "angle": 0.0,
                            "duration": 0.8,
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                        },
                        {
                            "type": "skill",
                            "skill_name": "climb_76_high_speed_down",
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                        },
                        {
                            "type": "locomotion",
                            "speed": running_speed,
                            "angle": 0.0,
                            "duration": 1.0,
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                        },
                        {
                            "type": "segment",
                            "speed": 0.0,
                            "angle": 0.0,
                            "duration": 2.5,
                            "vel_cmd_idx": 0,
                        },
                    ]
                configs.append({"name": f"{name}_{walk_duration}s_step{step_duration}s", "steps": steps})
    return configs, RUNNING_SPEEDUP


def low_speed_climb_94(walking_speed: float = WALKING_SPEED) -> tuple[list[dict[str, Any]], float]:
    scenarios_list = [("forward", 0.0), ("left_45", 45.0), ("left_90", 90.0), ("right_45", -45.0), ("right_90", -90.0)]
    walk_duration_list = [0.7, 0.9, 1.2, 1.5]
    step_duration_list = [0.5]

    stand_duration = 1.0

    configs = []
    for name, angular_velocity in scenarios_list:
        for walk_duration in walk_duration_list:
            walk_duration = round(walk_duration, 1)
            angle = angular_velocity * walk_duration
            for step_duration in step_duration_list:
                steps = [
                    {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": stand_duration, "vel_cmd_idx": 0},
                    {"type": "locomotion", "speed": walking_speed, "angle": 0.0, "duration": 1, "vel_cmd_idx": 1},
                    {
                        "type": "locomotion",
                        "speed": walking_speed,
                        "angle": angle,
                        "duration": walk_duration,
                        "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE[name],
                    },
                    {"type": "skill", "skill_name": "climb_94_up", "vel_cmd_idx": 1},
                    {
                        "type": "locomotion",
                        "speed": walking_speed,
                        "angle": 0.0,
                        "duration": step_duration,
                        "vel_cmd_idx": 1,
                    },
                    {"type": "skill", "skill_name": "climb_94_down", "vel_cmd_idx": 1},
                    {"type": "locomotion", "speed": walking_speed, "angle": 0.0, "duration": 0.5, "vel_cmd_idx": 1},
                    {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": 2.5, "vel_cmd_idx": 0},
                ]
                configs.append({"name": f"{name}_walk{walk_duration}s_step{step_duration}s", "steps": steps})
    return configs, WALKING_SPEEDUP


def high_speed_climb_94(running_speed: float = RUNNING_SPEED) -> tuple[list[dict[str, Any]], float]:
    scenarios_list = [
        ("run_forward", 0.0),
        ("run_left_45", 46.0),
        ("run_left_90", 90.0),
        ("run_right_45", -46.0),
        ("run_right_90", -90.0),
    ]
    walk_duration_list = [0.6, 1.0, 1.2]
    step_duration_list = [0]

    stand_duration = 1.0

    configs = []
    for name, angular_velocity in scenarios_list:
        for walk_duration in walk_duration_list:
            walk_duration = round(walk_duration, 1)
            angle = angular_velocity * walk_duration
            for step_duration in step_duration_list:
                steps = [
                    {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": stand_duration, "vel_cmd_idx": 0},
                    {
                        "type": "locomotion",
                        "speed": running_speed,
                        "angle": 0.0,
                        "duration": 1,
                        "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                    },
                    {
                        "type": "locomotion",
                        "speed": running_speed,
                        "angle": angle,
                        "duration": walk_duration,
                        "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE[name],
                    },
                    {
                        "type": "skill",
                        "skill_name": "climb_94_high_speed_up",
                        "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                    },
                ]
                if step_duration > 0:
                    steps.append(
                        {
                            "type": "locomotion",
                            "speed": running_speed,
                            "angle": 0.0,
                            "duration": step_duration,
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                        }
                    )

                steps.extend(
                    [
                        {
                            "type": "locomotion",
                            "speed": 0.5,
                            "angle": 0.0,
                            "duration": 0.8,
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                        },
                        {
                            "type": "skill",
                            "skill_name": "climb_94_high_speed_down",
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                        },
                        {
                            "type": "locomotion",
                            "speed": running_speed,
                            "angle": 0.0,
                            "duration": 1.0,
                            "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE["run_forward"],
                        },
                        {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": 2.5, "vel_cmd_idx": 0},
                    ]
                )
                configs.append({"name": f"{name}_{walk_duration}s_step{step_duration}s", "steps": steps})
    return configs, RUNNING_SPEEDUP


def low_speed_roll_78(walking_speed: float = WALKING_SPEED) -> tuple[list[dict[str, Any]], float]:
    steps = [
        {"type": "skill", "skill_name": "climb_76_up", "vel_cmd_idx": 1},
        {"type": "locomotion", "speed": walking_speed, "angle": 0.0, "duration": 1.0, "vel_cmd_idx": 1},
        {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": 0.8, "vel_cmd_idx": 1},
        {"type": "skill", "skill_name": "roll_78", "vel_cmd_idx": 11},
        {"type": "locomotion", "speed": walking_speed, "angle": 0.0, "duration": 1.0, "vel_cmd_idx": 1},
        {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": 2.5, "vel_cmd_idx": 0},
    ]
    return [{"name": "roll_78", "steps": steps}], WALKING_SPEEDUP


def low_speed_roll_90(walking_speed: float = WALKING_SPEED) -> tuple[list[dict[str, Any]], float]:
    walking_durations = [0.4, 1.0, 1.5]
    configs = []
    for walking_duration in walking_durations:
        steps = [
            {"type": "skill", "skill_name": "climb_94_up", "vel_cmd_idx": 1},
            {
                "type": "locomotion",
                "speed": walking_speed,
                "angle": 0.0,
                "duration": walking_duration,
                "vel_cmd_idx": 1,
            },
            {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": 0.8, "vel_cmd_idx": 1},
            {"type": "skill", "skill_name": "roll_90", "vel_cmd_idx": 11},
            {"type": "locomotion", "speed": walking_speed, "angle": 0.0, "duration": 1.0, "vel_cmd_idx": 1},
            {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": 2.5, "vel_cmd_idx": 0},
        ]
        configs.append({"name": f"roll_90_walk{walking_duration}s", "steps": steps})
    return configs, WALKING_SPEEDUP


def high_speed_roll_134(
    walking_speed: float = WALKING_SPEED, running_speed: float = RUNNING_SPEED
) -> tuple[list[dict[str, Any]], float]:
    scenarios_list = [
        ("run_forward", 0.0),
        ("run_left_45", 45.0),
        ("run_left_90", 90.0),
        ("run_right_45", -45.0),
        ("run_right_90", -90.0),
    ]

    walk_duration_list = [0.5, 0.8, 1.2, 1.5]
    step_duration_list = [1.0, 1.5]

    configs = []
    for name, angular_velocity in scenarios_list:
        for walk_duration in walk_duration_list:
            walk_duration = round(walk_duration, 1)
            angle = angular_velocity * walk_duration
            for step_duration in step_duration_list:
                steps = [
                    {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": 1.0, "vel_cmd_idx": 0},
                    {"type": "locomotion", "speed": running_speed, "angle": 0.0, "duration": 1, "vel_cmd_idx": 6},
                    {
                        "type": "locomotion",
                        "speed": running_speed,
                        "angle": angle,
                        "duration": walk_duration,
                        "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE[name],
                    },
                    {"type": "skill", "skill_name": "climb_134_up", "vel_cmd_idx": 6},
                    {
                        "type": "locomotion",
                        "speed": walking_speed,
                        "angle": 0.0,
                        "duration": step_duration,
                        "vel_cmd_idx": 6,
                    },
                    {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": 0.5, "vel_cmd_idx": 6},
                    {"type": "skill", "skill_name": "roll_136", "vel_cmd_idx": 6},
                    {"type": "locomotion", "speed": running_speed, "angle": 0.0, "duration": 1.0, "vel_cmd_idx": 6},
                    {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": 2.5, "vel_cmd_idx": 0},
                ]
                configs.append({"name": f"{name}_{walk_duration}s_step{step_duration}s", "steps": steps})
    return configs, RUNNING_SPEEDUP


def high_speed_hurdle(running_speed: float = RUNNING_SPEED) -> tuple[list[dict[str, Any]], float]:
    scenarios_list = [("run_forward", 0.0), ("run_left_45", 45.0), ("run_right_45", -45.0)]

    walk_duration_list = np.arange(0.3, 2.0, 0.1)

    configs = []
    for name, angular_velocity in scenarios_list:
        for walk_duration in walk_duration_list:
            walk_duration = round(walk_duration, 1)
            angle = angular_velocity * walk_duration
            steps = [
                {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": 1.0, "vel_cmd_idx": 0},
                {"type": "locomotion", "speed": running_speed, "angle": 0.0, "duration": 0.5, "vel_cmd_idx": 6},
                {
                    "type": "locomotion",
                    "speed": running_speed,
                    "angle": angle,
                    "duration": walk_duration,
                    "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE[name],
                },
                {"type": "skill", "skill_name": "hurdle", "vel_cmd_idx": 6},
                {"type": "locomotion", "speed": running_speed, "angle": 0.0, "duration": 0.5, "vel_cmd_idx": 6},
                {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": 2.5, "vel_cmd_idx": 0},
            ]
            configs.append({"name": f"{name}_{walk_duration}s", "steps": steps})
    return configs, RUNNING_SPEEDUP


def high_speed_dash_vault(running_speed: float = 2.0) -> tuple[list[dict[str, Any]], float]:
    scenarios_list = [("run_forward", 0.0), ("run_left_45", 45.0), ("run_right_45", -45.0)]

    walk_duration_list = np.arange(0.3, 2.0, 0.1)

    configs = []
    for name, angular_velocity in scenarios_list:
        for walk_duration in walk_duration_list:
            walk_duration = round(walk_duration, 1)
            angle = angular_velocity * walk_duration
            steps = [
                {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": 1.0, "vel_cmd_idx": 0},
                {"type": "locomotion", "speed": running_speed, "angle": 0.0, "duration": 0.5, "vel_cmd_idx": 6},
                {
                    "type": "locomotion",
                    "speed": running_speed,
                    "angle": angle,
                    "duration": walk_duration,
                    "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE[name],
                },
                {"type": "skill", "skill_name": "dash_vault", "vel_cmd_idx": 6},
                {"type": "locomotion", "speed": running_speed + 0.3, "angle": 0.0, "duration": 0.5, "vel_cmd_idx": 6},
                {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": 2.5, "vel_cmd_idx": 0},
            ]
            configs.append({"name": f"{name}_{walk_duration}s", "steps": steps})
    return configs, RUNNING_SPEEDUP


def high_speed_speed_vault(running_speed: float = RUNNING_SPEED) -> tuple[list[dict[str, Any]], float]:
    scenarios_list = [
        ("run_forward", 0.0),
        ("run_left_45", 45.0),
        ("run_left_90", 90.0),
        ("run_right_45", -45.0),
        ("run_right_90", -90.0),
    ]
    walk_duration_list = [0.3, 0.5, 0.95, 1.2]
    configs = []
    for name, angular_velocity in scenarios_list:
        for walk_duration in walk_duration_list:
            walk_duration = round(walk_duration, 1)
            angle = angular_velocity * walk_duration
            steps = [
                {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": 1.0, "vel_cmd_idx": 0},
                {"type": "locomotion", "speed": running_speed, "angle": 0.0, "duration": 1, "vel_cmd_idx": 6},
                {
                    "type": "locomotion",
                    "speed": running_speed,
                    "angle": angle,
                    "duration": walk_duration,
                    "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE[name],
                },
                {"type": "skill", "skill_name": "speed_vault", "vel_cmd_idx": 6},
                {"type": "locomotion", "speed": running_speed, "angle": 0.0, "duration": 0.5, "vel_cmd_idx": 6},
                {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": 2.5, "vel_cmd_idx": 0},
            ]
            configs.append({"name": f"{name}_{walk_duration}s", "steps": steps})
    return configs, RUNNING_SPEEDUP


def high_speed_cat_vault(running_speed: float = RUNNING_SPEED) -> tuple[list[dict[str, Any]], float]:
    scenarios_list = [("run_forward", 0.0), ("run_left_45", 45.0), ("run_right_45", -45.0)]
    walk_duration_list = np.arange(0.3, 2.0, 0.1)
    configs = []
    for name, angular_velocity in scenarios_list:
        for walk_duration in walk_duration_list:
            walk_duration = round(walk_duration, 1)
            angle = angular_velocity * walk_duration
            steps = [
                {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": 1.0, "vel_cmd_idx": 0},
                {"type": "locomotion", "speed": running_speed, "angle": 0.0, "duration": 0.5, "vel_cmd_idx": 6},
                {
                    "type": "locomotion",
                    "speed": running_speed,
                    "angle": angle,
                    "duration": walk_duration,
                    "vel_cmd_idx": VEL_CMD_MAPPING_REVERSE[name],
                },
                {"type": "skill", "skill_name": "cat_vault", "vel_cmd_idx": 6},
                {"type": "locomotion", "speed": running_speed, "angle": 0.0, "duration": 0.5, "vel_cmd_idx": 6},
                {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": 2.5, "vel_cmd_idx": 0},
            ]
            configs.append({"name": f"{name}_{walk_duration}s", "steps": steps})
    return configs, RUNNING_SPEEDUP


def high_speed_jump_35(running_speed: float = RUNNING_SPEED) -> tuple[list[dict[str, Any]], float]:
    steps = [
        {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": 1.0, "vel_cmd_idx": 0},
        {"type": "skill", "skill_name": "step_high_speed_up", "vel_cmd_idx": 6},
        {"type": "locomotion", "speed": running_speed, "angle": 0.0, "duration": 1.2, "vel_cmd_idx": 6},
        {"type": "skill", "skill_name": "jump_35", "vel_cmd_idx": 6},
        {"type": "skill", "skill_name": "jump_35", "vel_cmd_idx": 6},
        {"type": "locomotion", "speed": running_speed, "angle": 0.0, "duration": 0.8, "vel_cmd_idx": 6},
        {"type": "skill", "skill_name": "step_high_speed_down", "vel_cmd_idx": 6},
        {"type": "locomotion", "speed": running_speed, "angle": 0.0, "duration": 0.2, "vel_cmd_idx": 6},
        {"type": "segment", "speed": 0.0, "angle": 0.0, "duration": 2.5, "vel_cmd_idx": 0},
    ]
    return [{"name": "jump_35", "steps": steps}], RUNNING_SPEEDUP
