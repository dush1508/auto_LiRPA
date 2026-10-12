import argparse
import os

# examples/vision/ -> examples/ -> auto_LiRPA-master/ -> the harness checkout holding nets/.
_HARNESS_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             os.pardir, os.pardir, os.pardir))

DEFAULT_NETS_DIR = os.path.abspath(
    os.environ.get("CF_NETS_ROOT") or os.path.join(_HARNESS_ROOT, "nets"))
# Prefer the shared dataset; also support the existing ConstraintFlow download.
# An explicit override is authoritative, even when the path is missing.
_tinyimagenet_candidates = [
    os.path.join(_HARNESS_ROOT, "tinyimagenet", "tiny-imagenet-200"),
    os.path.join(os.environ.get("CF_GPU_DIR") or os.path.join(_HARNESS_ROOT, "constraintflow-gpu"),
                 "tinyimagenet", "tiny-imagenet-200"),
]
DEFAULT_TINYIMAGENET_DIR = os.path.abspath(
    os.environ.get("CF_TINYIMAGENET_ROOT")
    or next((path for path in _tinyimagenet_candidates
             if os.path.isdir(os.path.join(path, "train"))), _tinyimagenet_candidates[0]))


def parse_args():
    parser = argparse.ArgumentParser(description="ConstraintFlow experiments")

    parser.add_argument('--single_experiment', dest='single_experiment', action='store_true', help='Enable single experiment mode')
    parser.add_argument('--multiple_experiments', dest='single_experiment', action='store_false', help='Disable single experiment mode')
    parser.set_defaults(single_experiment=True)

    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--eps', type=float, default=0.01, help='Epsilon')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset')

    parser.add_argument('--network', type=str, default='mnist_relu_3_50', help='Network name')
    parser.add_argument('--network_format', type=str, default='onnx', help='Network format')
    parser.add_argument('--network_type', type=str, default='fcn', help='Network type')
    parser.add_argument('--nets_dir', type=str, default=DEFAULT_NETS_DIR,
                         help='Root folder containing mnist/ and cifar/ subfolders of onnx networks '
                              f'(default: {DEFAULT_NETS_DIR})')

    parser.add_argument('--program', type=str, default='deeppoly', help='AutoLirpa program')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='cpu',
                         help='Device to run bound computation on: cpu or gpu (CUDA)')
    
    parser.add_argument('--num_runs', type=int, default=1, help='Run the experiment n times for average times')
    parser.add_argument('--warmup', type=int, default=0, help='Number of warmup runs on different data before the timed run')
    parser.add_argument('--repeat', type=int, default=1, help='Number of timed runs, each reported separately. Independent of --warmup: the warmup runs (if any) still happen once, before the first timed run.')
    parser.add_argument('--num_images', type=int, default=0, help='Verify this many test images, batch_size at a time, one timed run per batch (batch i = images [i*batch_size, (i+1)*batch_size)); warmup j uses the images right after them. Requires --results_file; replaces --repeat. 0 (default): the --repeat protocol.')
    parser.add_argument('--results_file', type=str, default=None, help='With --num_images: write one JSON line per warmup and per timed batch (time, peak/resident memory, bounds) here, outside the timer.')
    parser.add_argument('--perturb_eps', type=float, default=0.0, help="Std of iid Gaussian noise added to the network's weights independently before each warmup and each timed run, to test whether weight values (not just shapes) affect measured runtime. 0 (default) disables perturbation and every run uses the unmodified network.")
    parser.add_argument('--perturb_seed', type=int, default=0, help="Seed for --perturb_eps's noise, for reproducible sweeps.")
    parser.add_argument('--exp_name', type=str, default='eran')

    parser.add_argument('--print_intermediate_results', dest='print_intermediate_results', action='store_true', help='Print intermediate results')
    parser.set_defaults(print_intermediate_results=False)

    parser.add_argument('--record_time', dest='record_time', action='store_true', help='Log function times')
    parser.set_defaults(record_time=False)

    parser.add_argument('--logs', type=str, default='stdout', help='Logs file')
    #FEAT: --dtype float64 runs the analysis in fp64, the precision oracle for bound mismatches.
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float32', help='float64 computes every float tensor in fp64 (a precision oracle for bound mismatches).')
    #FEAT: --profile_comm records CPU-GPU communication per timed batch (comm_profile.py).
    parser.add_argument('--profile_comm', action='store_true', help='With --num_images: profile each timed batch with torch.profiler and add syncs, alloc_syncs, gpu_idle_pct and sync_calls to its results line; its time goes to profiled_time_s instead of time_s. Warmups are not profiled.')


    args = parser.parse_args()
    return args