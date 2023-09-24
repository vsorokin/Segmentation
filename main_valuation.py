import arguments
import logging
import logging_config
import serialization
import image_pipeline
import numpy as np
from sewar.full_ref import rmse
import seg_metrics.seg_metrics as sg

classes = [0, 1, 2, 3]  # exclude background?
negative_metrics = frozenset(['hd', 'hd95', 'rmse', 'msd', 'mdsd', 'fpr', 'fnr', 'stdsd'])

full_names = {
    "dice": "Dice score",
    "jaccard": "Jaccard index",
    "recall": "Sensitivity",
    "Specificity": "Specificity",
    "precision": "Precision",
    "hd": "Hausdorff distance",
    "hd95": "Hausdorff distance (P95\%)",
    "msd": "Mean surface distance",
    "mdsd": "Median surface distance",
    "vs": "Volumetric similarity",
}


def main():
    logging_config.configure_logging(log_to_stdout=True)

    # result_path="/Users/sw/work/msc_ai_diss/output/2023_09_12__07_53_53_val"
    # result_path = "/Users/sw/work/msc_ai_diss/output/2023_09_12__00_11_18_val"
    result_path = "/Users/sw/work/msc_ai_diss/output/2023_09_13__05_59_35_val"
    # result_path = "/Users/sw/work/msc_ai_diss/output/2023_09_14__00_22_01_val"
    # result_path = "/Users/sw/work/msc_ai_diss/output/2023_09_15__05_37_53_val"
    # result_path = "/Users/sw/work/msc_ai_diss/output/2023_09_15__12_52_22_val"
    ser = serialization.Serialization(result_path)

    all_metrics_original = []
    all_metrics_restored = []
    sample_indices = [i for i in range(10)]
    for sample_idx in sample_indices:
        ip = ser.read_from_file(f"ip_{sample_idx}.pkl")

        metrics_original = calculate_metrics(ip.segmentation_3d, ip.original_pred_segmentation_3d)
        metrics_original['rmse'] = calc_rmse_for_classes(ip.segmentation_3d,
                                                         ip.original_pred_segmentation_3d)
        metrics_original['Specificity'] = calc_specificity(metrics_original)
        all_metrics_original.append(metrics_original)

        metrics_restored = calculate_metrics(ip.segmentation_3d, ip.avg_segmentation_restored_3d)
        metrics_restored['rmse'] = calc_rmse_for_classes(ip.segmentation_3d,
                                                         ip.avg_segmentation_restored_3d)
        metrics_restored['Specificity'] = calc_specificity(metrics_restored)
        all_metrics_restored.append(metrics_restored)

    m_original = accumulate_metrics(all_metrics_original)
    m_restored = accumulate_metrics(all_metrics_restored)

    m_avg_original = calculate_avg_metrics(m_original)
    m_avg_restored = calculate_avg_metrics(m_restored)

    for i, sample_idx in enumerate(sample_indices):
        print(f"Original {sample_idx}")
        orig = read_metrics(all_metrics_original[i])
        print_metrics(orig)
        print(f"Restored {sample_idx}")
        print_metrics(read_metrics(all_metrics_restored[i]), base=orig)
        print("-------------------------------------------------------------")

    print("AVERAGE")
    print("Original")
    print_metrics(m_avg_original)
    print("Restored")
    print_metrics(m_avg_restored, base=m_avg_original)


def calc_specificity(metrics):
    r = []
    for v in metrics['fpr']:
        r.append(1.0 - v)
    return r


def calc_rmse_for_classes(GT, P):
    result = []
    for c in classes:
        result.append(rmse(GT == c, (P == c).numpy()))
    return result


def accumulate_metrics(all_metrics_from_file):
    result = {}
    for m in all_metrics_from_file[0].keys():
        result[m] = {}
        for c in classes:
            result[m][c] = []

    for mm in all_metrics_from_file:
        for m in mm.keys():
            for i, c in enumerate(mm["label"]):
                result[m][c].append(mm[m][i])

    return result


def read_metrics(all_metrics_from_file):
    result = {}
    for m in all_metrics_from_file.keys():
        result[m] = {}

    assert all_metrics_from_file["label"] == classes
    for m in all_metrics_from_file.keys():
        for i, c in enumerate(all_metrics_from_file["label"]):
            result[m][c] = all_metrics_from_file[m][i]

    return result


def calculate_avg_metrics(metrics):
    result = {}
    for m in metrics.keys():
        result[m] = {}

    for m, per_class in metrics.items():
        for c, values in per_class.items():
            result[m][c] = np.mean(values)

    return result


def print_metrics(metrics, base=None):
    for m, caption in full_names.items():
        print(f"{caption} & ", end="")
        for i, c in enumerate(classes):
            diff_rel_perc_str = None
            change_is_good = False
            if base is None:
                sign = None
            else:
                diff = metrics[m][c] - base[m][c]
                diff_rel = diff / abs(base[m][c]) if base[m][c] != 0. else 0.
                diff_rel_perc_str = f"{diff_rel * 100:+.1f}\\%"
                if diff_rel_perc_str == "+0.0\\%" or diff_rel_perc_str == "-0.0\\%":
                    diff_rel_perc_str = "0.0\\%"
                    change_is_good = False
                    sign = "🟡"
                else:
                    change_is_good = (m in negative_metrics) ^ (diff > 0)
                    sign = "🟢" if change_is_good else "🔴"
            if (diff_rel_perc_str is not None):
                if change_is_good:
                    parts = diff_rel_perc_str.split(".")
                    print("\\textbf{%s}&\\textbf{%s} " % (parts[0], parts[1]), end="")
                else:
                    diff_rel_perc_str_tex = diff_rel_perc_str.replace('.', '&')
                    print(f"{diff_rel_perc_str_tex} ", end="")

                if i < 3:
                    print(" & ", end="")

        print(r" \\")


def calculate_metrics(GT, P):
    assert len(P.shape) == 3
    metrics = sg.write_metrics(labels=classes,
                               gdth_img=GT,
                               pred_img=P.numpy(),
                               csv_file="metrics.tmp")
    assert len(metrics) == 1
    return metrics[0]


if __name__ == '__main__':
    main()
