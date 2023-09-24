"""
Runs training.
"""
import logging
import sys
from training import Training
import arguments
import logging_config


def main():
    args = arguments.parse_args()
    result_path = logging_config.get_result_path(args.output_dir)
    logging_config.configure_logging(result_path, loglevel=args.loglevel.upper(),
                                     log_to_stdout=args.log_to_stdout)

    unet_filters = 30
    metadata = {"command_line": f"{sys.argv}",
                "result_path": result_path,
                "parsed_args": vars(args),
                "unet_filters": unet_filters}
    save_metadata(result_path, metadata)

    logging.info(f"RESULT PATH: [{result_path}]")

    training = Training(metadata,
                        result_path,
                        device=args.device,
                        brats_dir=args.brats_dir,
                        brats_validation_dir=args.brats_validation_dir,
                        batch_size=args.batch_size,
                        use_cache=args.use_cache,
                        cache_ratio=args.cache_ratio,
                        use_cache_for_validation=args.use_cache_for_validation,
                        data_loading_num_workers=args.data_loading_num_workers,
                        checkpoint_to_load=args.checkpoint_to_load,
                        epochs=args.epochs,
                        learning_rate=args.learning_rate,
                        momentum=args.momentum,
                        epochs_per_checkpoint=args.epochs_per_checkpoint,
                        epochs_per_permanent_checkpoint=args.epochs_per_permanent_checkpoint,
                        unet_filters=unet_filters)
    training.run()


def save_metadata(result_path, metadata):
    with open(f"{result_path}/metadata.txt", "w") as metadata_file:
        for k, v in metadata.items():
            metadata_file.write(f"{k}: {v}\n")


if __name__ == '__main__':
    main()
