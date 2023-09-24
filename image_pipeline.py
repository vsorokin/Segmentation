import pipeline

import logging
import torch
import data_processing
import seg_metrics.seg_metrics as sg


class ImagePipeline:
    csv_file = 'metrics.tmp'
    # metrics_to_calculate = ['dice', 'hd', 'hd95', 'precision', 'recall']
    classes = [1, 2, 3]  # exclude background

    def __init__(self, mri_4d, segmentation_4d, inference, transformation_count, device, alpha,
                 sigma, order=1, restore_order=1, image_clipped=True, keep_pipelines=False):
        assert len(mri_4d.shape) == 4
        assert len(segmentation_4d.shape) == 4
        assert segmentation_4d.shape[0] == 1

        self.mri_4d = mri_4d
        self.segmentation_3d = segmentation_4d[0]
        self.inference = inference
        self.device = device

        logging.info("Transformation, inference, restore...")
        pipelines = []
        for i in range(transformation_count):
            logging.info(f"Processing transformation #{i + 1} of {transformation_count}")
            tp = pipeline.TransformationPipeline(mri_4d, self.segmentation_3d, inference,
                                                 device=device, alpha=alpha, sigma=sigma,
                                                 order=order, restore_order=restore_order,
                                                 image_clipped=image_clipped)
            pipelines.append(tp)

        logging.info("Deriving restored segmentation...")
        all_pred_restored_4d = torch.stack([p.pred_restored_4d for p in pipelines])
        if keep_pipelines:
            self.pipelines = pipelines
        avg_pred_restored_4d, _ = torch.median(all_pred_restored_4d, dim=0)
        self.avg_segmentation_restored_3d = data_processing.segmentation_3d_from_prediction_4d(
            avg_pred_restored_4d)

        self.original_pred_segmentation_3d = self.calculate_original_pred_segmentation_3d()

    def calculate_original_pred_segmentation_3d(self):
        logging.info("Inference: original")
        dataset = data_processing.sample_to_dataset(self.mri_4d, device=self.device)
        pred_batch_5d = self.inference.predict(dataset)
        return data_processing.segmentation_3d_from_prediction_4d(pred_batch_5d[0])
