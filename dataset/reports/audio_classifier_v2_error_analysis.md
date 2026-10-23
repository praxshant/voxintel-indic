# Direct Audio Error Analysis

This is descriptive analysis of a frozen final-test report. It must not be used to retune the evaluated model.

## Robustness summary

- Accuracy on accepted predictions: 69.87%
- Coverage: 96.15%
- False-accept rate for `other`: 36.56%

## Conditions

| Condition | Accuracy | Coverage | Other false accepts |
|---|---:|---:|---:|
| gaussian_snr10 | 58.95% | 93.08% | 44.58% |
| traffic_snr15 | 67.55% | 96.41% | 49.40% |
| telephone_channel | 71.24% | 97.18% | 36.78% |
| gaussian_snr20 | 72.22% | 96.92% | 34.83% |
| unseen_combined | 73.02% | 96.92% | 26.14% |
| office_snr15 | 75.80% | 96.41% | 28.74% |

## Lowest-recall labels

| Label | Support | Precision | Recall |
|---|---:|---:|---:|
| balance | 180 | 56.13% | 48.33% |
| complaint_status | 180 | 81.51% | 53.89% |
| update_details | 180 | 78.83% | 60.00% |
| other | 540 | 91.78% | 62.04% |
| block_card | 180 | 62.98% | 63.33% |

## Most common confusions

| Actual | Predicted | Count |
|---|---|---:|
| other | emi_due | 58 |
| other | pay_bill | 36 |
| other | branch_timing | 26 |
| other | block_card | 25 |
| complaint_status | balance | 23 |
| emi_due | pay_bill | 22 |
| balance | pay_bill | 21 |
| balance | block_card | 20 |
| balance | branch_timing | 20 |
| transfer_to_agent | pay_bill | 20 |

## Interpretation

- Use the weakest conditions and labels to guide qualitative inspection, not final-test retuning.
- The `other` false-accept rate remains the principal safety limitation of this direct-audio probe.
- The next model decision must be selected on the frozen calibration manifest before another final-test run.
