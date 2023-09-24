import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--brats_dir", required=True)
    parser.add_argument("--brats_validation_dir", required=False)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", dest="checkpoint_to_load", required=False)
    parser.add_argument("--epochs", type=int, required=False, default=5)
    parser.add_argument("--epochs_per_checkpoint", type=int, required=False, default=5)
    parser.add_argument("--epochs_per_permanent_checkpoint", type=int, required=False, default=50)
    parser.add_argument("--device", required=False, default='cuda')
    parser.add_argument("--batch_size", type=int, required=False, default=1)
    parser.add_argument("--learning_rate", type=float, required=False, default=0.01)
    parser.add_argument("--momentum", type=float, required=False, default=0.9)
    parser.add_argument("--data_loading_num_workers", type=int, required=False, default=0)
    parser.add_argument("--loglevel", required=False, default="info")
    parser.add_argument("--use_cache", type=bool, required=False, default=False)
    parser.add_argument("--cache_ratio", type=float, required=False, default=1.0)
    parser.add_argument("--use_cache_for_validation", type=bool, required=False, default=True)
    parser.add_argument("--log_to_stdout", type=bool, required=False, default=True)
    args = parser.parse_args()
    print(f"ARGUMENTS: {args}")
    return args
