import elastic_transform
import visualisation
import data_processing as dataproc
import logging
import loss


class TransformationPipeline:
    def __init__(self, mri_4d, segmentation_3d, inference, device, alpha, sigma, order=1,
                 restore_order=1,
                 image_clipped=True,
                 noop=False):
        self.image_shape = (240, 240, 144 if image_clipped else 155)

        assert len(mri_4d.shape) == 4
        assert mri_4d[0].shape == self.image_shape
        assert len(segmentation_3d.shape) == 3
        assert segmentation_3d.shape == self.image_shape

        self.mri_4d = mri_4d
        self.target_segmentation_3d = segmentation_3d

        self.grid_3d = visualisation.create_grid_3d(self.image_shape)
        if not noop:
            self.transform = elastic_transform.ElasticTransformation3D(alpha=alpha, sigma=sigma,
                                                                       order=order,
                                                                       image_clipped=True)

        # transform MRI
        logging.info("Transforming...")
        if noop:
            self.mri_transformed_4d = mri_4d
            self.grid_transformed_3d = self.grid_3d
        else:
            self.mri_transformed_4d = self.transform.transform_4d(mri_4d)
            self.grid_transformed_3d = self.transform.transform_3d(self.grid_3d)

        # predict from the transformed MRI
        logging.info("Inference...")
        dataset = dataproc.sample_to_dataset(self.mri_transformed_4d, device=device)
        pred_transformed = inference.predict(dataset)
        pred_transformed_4d = pred_transformed[0]

        # derive transformed segmentation from transformed prediction
        self.pred_segmentation_transformed_3d = dataproc.segmentation_3d_from_prediction_4d(
            pred_transformed_4d)

        # restore the transformed prediction
        logging.info("Restoring...")
        if noop:
            self.pred_restored_4d = pred_transformed_4d
            # These are for visualisations only
            self.mri_restored_4d = mri_4d
            self.grid_restored_3d = self.grid_transformed_3d
        else:
            self.pred_restored_4d = self.transform.restore_4d(pred_transformed_4d,
                                                              order=restore_order)
            # These are for visualisations only
            self.mri_restored_4d = self.transform.restore_4d(mri_4d, order=restore_order)
            self.grid_restored_3d = self.transform.restore_3d(self.grid_transformed_3d,
                                                              order=restore_order)

        # derive segmentation from restored prediction
        self.pred_segmentation_restored_3d = dataproc.segmentation_3d_from_prediction_4d(
            self.pred_restored_4d)

        self.dice_loss = loss.DiceLoss(device="cpu")

    def restored_loss(self):
        return self.dice_loss.calculate_dice_loss(self.pred_restored_4d,
                                                  self.target_segmentation_3d)
