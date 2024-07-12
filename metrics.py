def compute_RMS(
    data: np.ndarray,
    axis: int,
    mean_based_on_estimated_data: bool,
) -> float:
    """
    Computes a root mean square value.
    """
    if mean_based_on_estimated_data: # when computing the RMSE based on an estimated mean value, i.e. NOT based on actual data mean value
        return float(
            np.sqrt(np.sum(np.square(data)) / (data.shape[axis] - 1))
        )
    else:
        return float(np.sqrt(np.mean(np.square(data), axis=axis)))


def compute_RMSEs_over_all_samples(
    prediction: np.ndarray, ground_truth: np.ndarray
) -> dict:
    """
    This function computes the signals'/time series' RMSE.

    0-axis sample index, 1-axis time-axis.
    """
    list_sample_rmse_values = [None] * prediction.shape[0]

    idx_sample = 0
    for predicted_sample, sample_ground_truth in zip(
        prediction, ground_truth
    ):
        list_sample_rmse_values[idx_sample] = compute_RMS(
            data=(predicted_sample - sample_ground_truth),
            axis=0,
            mean_based_on_estimated_data=False,
        )
        idx_sample += 1

    return compute_min_max_median(list_sample_rmse_values)


def compute_RMS_of_RMSE_of_dataset(
    dataset_prediction: np.ndarray, dataset_ground_truth: np.ndarray
) -> float:
    """
    Computes the root mean square of a time-series dataset's RMSE values.
    """
    sample_rmse_values = compute_RMSEs_over_all_samples(
        prediction=dataset_prediction, ground_truth=dataset_ground_truth
    )["list_samples"]

    rmse_dataset = compute_RMS(
        data=np.array(sample_rmse_values), axis=0, mean_based_on_estimated_data=False
    )

    return rmse_dataset


def compute_min_max_median(list_samples: list) -> dict:
    """
    Computes minimum, maximum and median value.
    """
    max_val = float(np.max(list_samples))
    idx_max = int(np.argmax(list_samples))

    min_val = float(np.min(list_samples))
    idx_min = int(np.argmin(list_samples))

    list_original_sample_indices = [x for x in range(0, len(list_samples))]
    zipped_lists = list(zip(list_samples, list_original_sample_indices))

    sorted_zipped_lists = sorted(zipped_lists, key=lambda x: float(x[0]))

    median_val = float(sorted_zipped_lists[len(list_samples) // 2][0])
    idx_median = sorted_zipped_lists[len(list_samples) // 2][1]

    return {
        "list_samples": list_samples,
        "max_val": max_val,
        "idx_max_val": idx_max,
        "min_val": min_val,
        "idx_min_val": idx_min,
        "median_val": median_val,
        "idx_median_val": idx_median,
    }


def compute_avg_signal(dataset: np.ndarray) -> np.ndarray:
    """
    Average signal/time series over all dataset samples.
    """
    return np.mean(dataset, axis=0)


def compute_dataset_RMS_wrt_avg_signal(ground_truth: np.ndarray) -> float:
    """
    This function computes the RMS of the variation of the dataset.
    This measure is essentially the RMS of the difference between individual samples and the average signal.

    The result depends on the ground truth data only, being a measure for
    the spread of the data as compared to the average signal.
    """
    avg_signal = compute_avg_signal(ground_truth)

    number_of_samples = ground_truth.shape[0]

    stacked_avg_signal = np.vstack([avg_signal] * number_of_samples)

    sample_rmse_values = compute_RMSEs_over_all_samples(
        prediction=stacked_avg_signal,
        ground_truth=ground_truth,
    )["list_samples"]

    rmse_dataset = compute_RMS(
        data=np.array(sample_rmse_values),
        axis=0,
        mean_based_on_estimated_data=False,
    )

    return rmse_dataset


def compute_rel_var_RMSE_for_single_sample(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    idx_ground_truth_sample: int,
) -> float:
    """
    This metric quantifies the error of a single predicted sample in relation to variation of dataset.

    0-axis: sample index, 1-axis: time-axis.
    """
    rmse_sample = compute_RMS(
        data=(
            prediction - ground_truth[idx_ground_truth_sample]
        ),
        axis=0,
        mean_based_on_estimated_data=False,
    )
    rms_var_dataset = compute_dataset_RMS_wrt_avg_signal(ground_truth)

    return rmse_sample / rms_var_dataset


def compute_rel_var_RMSE_for_each_sample(
    prediction: np.ndarray, ground_truth: np.ndarray
) -> dict:
    """
    Computes the rel var RMSE computation for each sample.

    0-axis sample index,  1-axis time-axis.
    """
    rmse_ground_truth = compute_RMS_of_variation_of_dataset(dataset_ground_truth)

    dict_rmse_results = compute_RMSEs_over_all_samples(
        prediction=prediction, ground_truth=ground_truth
    )

    list_all_rel_var_RMSEs = np.divide(
        dict_rmse_results["list_samples"], rmse_ground_truth
    )

    return compute_min_max_median(list_all_rel_var_RMSEs)



def compute_rel_var_RMSE_for_dataset(
    prediction: np.ndarray, ground_truth: np.ndarray
) -> float:
    """
    The rel var RMSE for dataset is the quadratically weighted average error of the predicted samples w.r.t. the variation of the dataset.

    0-axis sample index,  1-axis time-axis.
    """
    list_rel_var_RMSE_all_samples = compute_rel_var_RMSE_for_each_sample(
        prediction=prediction, ground_truth=ground_truth
    )["list_samples"]

    rms_dataset = compute_RMS(
        data=np.array(list_rel_var_RMSE_all_samples),
        axis=0,
        mean_based_on_estimated_data=False,
    )

    return rms_dataset


