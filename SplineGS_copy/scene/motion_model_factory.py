from scene.deterministic_motion_model import DeterministicMotionModelCZ
from scene.fm_motion_model import FMMotionModelCZ
from scene.mlp_motion_model import MLPMotionModelCZ
from scene.spline_motion_model import SplineMotionModelCZ


def build_motion_model(args):
    motion_model_type = getattr(args, "motion_model_type", "spline").lower()
    builders = {
        "spline": lambda: SplineMotionModelCZ(args.control_num),
        "deterministic": lambda: DeterministicMotionModelCZ(args.control_num),
        "fm": lambda: FMMotionModelCZ(args.control_num),
        "mlp": lambda: MLPMotionModelCZ(
            args.control_num,
            hidden_dim=getattr(args, "motion_hidden_dim", 128),
            num_layers=getattr(args, "motion_num_layers", 3),
            pe_freqs=getattr(args, "motion_pe_freqs", 4),
        ),
    }
    if motion_model_type not in builders:
        raise ValueError(f"Unsupported motion model type: {motion_model_type}")
    return builders[motion_model_type]()
