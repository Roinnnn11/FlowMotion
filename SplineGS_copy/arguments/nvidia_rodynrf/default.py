OptimizationParams = dict(
    iterations=12500,
    coarse_batch_size=4,   # A800: was 2
    fine_batch_size=4,     # A800: was 2
    coarse_iterations=5_000,
    static_iterations=5_000,
    densify_from_iter=500,
    densify_until_iter=12000,
    opacity_reset_interval=3_000,
    densify=1,
    stat_npts=50000,       # A800: was 20000
    dyn_npts=30000,        # A800: was 10000
    desicnt=12,
)