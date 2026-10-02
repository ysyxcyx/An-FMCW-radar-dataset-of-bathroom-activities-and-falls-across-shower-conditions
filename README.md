# RD/RA Action Classification and Fall Recognition Example

`example_action_fall_classifier.py` demonstrates two classification tasks using the processed radar data in `TOP_SPRAY`:

- **Action classification:** distinguish the 14 activity, posture, and background categories recognized by the label reader.
- **Fall recognition:** classify each sampled interval as `Fall down` or `Non-fall`.

`Fall down and get up` describes rising after a fall. It is a separate category in action classification and belongs to `Non-fall` in the binary task.

## Requirements

The example runs on a CPU using Python, NumPy, h5py, and Matplotlib. It was run with Python 3.13. No GPU or deep-learning framework is required.

```bash
python -m pip install numpy h5py matplotlib
```

## Input data

Set `--data-root` to the release directory containing `TOP_SPRAY`. Each included session requires:

```text
TOP_SPRAY/
  <session_id>/
    RD.mat
    RA.mat
    label.csv
```

RD contains complex range-Doppler coefficients and RA contains linear range-angle power. Their MATLAB dimensions are `[51, 128, N]` and `[51, 33, N]`, respectively. The script reads frame windows directly from the HDF5 files and restores the axis order.

The current label reader expects the five-column VIA export with `temporal_segment_start`, `temporal_segment_end`, and JSON `metadata`. It accepts category names under either `Activity` or `TEMPORAL-SEGMENTS`. It normalizes `No perosn` to `No person` and `bend posture` to `Bend posture` in memory. Unrecognized or malformed rows and sessions without readable labels are skipped. The VIA `temporal_coordinates` export layout is not supported by this example.

## Run

Run the following command from this `examples` directory:

```bash
python example_action_fall_classifier.py --data-root "F:\FALL_RD_RA_RELEASE\data_v2" --group TOP_SPRAY --out-dir action_fall_results
```

On Windows, `py -3.13` can replace `python`. For another computer, replace the data path with the local release directory. Display all options with:

```bash
python example_action_fall_classifier.py --help
```

## Processing and network

1. Create one sample per recognized annotation interval. Convert its midpoint from video seconds to a zero-based radar frame index using `round(midpoint_seconds * 20)`.
2. Read a 16-frame window around that center, corresponding to 0.8 seconds at 20 frames/s. Centers are clamped to the recording bounds, and edge windows are shifted to fit the recording.
3. Compute `log1p(abs(RD))` and `log1p(max(RA, 0))`.
4. Average-pool RD to `13 x 16` cells and RA to `13 x 9` cells per frame. For the released array dimensions, the fast pooling path uses range bins 0-38, all 128 Doppler bins, and the first 27 angle bins (-64 to 40 degrees).
5. Concatenate the temporal mean and standard deviation of both pooled representations into a 650-element feature vector.
6. Split samples by session and standardize features using training-set statistics.
7. Train separate networks for the two tasks: `650 -> Linear(48) -> ReLU -> Linear(C)`, where C is the number of classes. Training uses class-weighted softmax cross-entropy and mini-batch stochastic gradient descent.

Default settings:

| Parameter | Value |
| --- | --- |
| Test session fraction | 0.25 |
| Window length | 16 frames |
| Hidden units | 48 |
| Epochs | 35 |
| Batch size | 64 |
| Learning rate | 0.002 |
| Random seed | 20261002 |

The split keeps each session in one partition. If a category is absent from the test set, the script moves a session containing that category into the test set; the final test fraction may therefore exceed the requested value. Participants may occur in both partitions. The exact session lists are saved in `run_summary.json`.

## Outputs

The output directory contains:

| File | Contents |
| --- | --- |
| `action_classification_confusion_matrix.png` | Action-classification confusion matrix |
| `fall_detection_confusion_matrix.png` | Binary fall-recognition confusion matrix |
| `*_confusion_matrix.csv` | Raw matrix counts; rows are true labels and columns are predicted labels |
| `*_metrics.json` | Accuracy, macro-F1, per-class precision/recall/F1, class IDs, and training loss |
| `*_predictions.csv` | True and predicted labels for test samples |
| `*_model.npz` | Network weights and biases |
| `feature_standardization.npz` | Training-set mean and standard deviation for raw features |
| `run_summary.json` | Session split, sample counts, task definitions, and metrics |

CSV matrix axes follow the numeric order in `class_to_id` in the corresponding metrics JSON. For model reuse, apply the mean and standard deviation from `feature_standardization.npz` to raw features; the additional `mean` and `std` fields inside the model files describe already-standardized training features.

## Evaluation scope

This example evaluates classification of windows selected using known annotation intervals. A window can include neighboring actions when an interval is shorter than 16 frames. The reported metrics describe this interval-centered task; continuous-stream fall detection requires a sliding-window evaluation. Check timestamp coverage before using the results as a benchmark, because the current implementation clamps out-of-range centers to recording boundaries.

The script reads the release files and writes results to `--out-dir`; it does not modify radar arrays or labels.

