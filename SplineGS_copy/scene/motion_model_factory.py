from scene.deterministic_motion_model import DeterministicMotionModelCZ
from scene.fm_motion_model import FMMotionModelCZ
from scene.spline_motion_model import SplineMotionModelCZ


def build_motion_model(args):
    motion_model_type = getattr(args, "motion_model_type", "spline").lower()
    builders = {
        "spline": SplineMotionModelCZ,
        "deterministic": DeterministicMotionModelCZ,
        "fm": FMMotionModelCZ,
    }
    if motion_model_type not in builders:
        raise ValueError(f"Unsupported motion model type: {motion_model_type}")
    return builders[motion_model_type](args.control_num)
