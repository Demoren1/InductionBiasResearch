# Robustness and two-layer results

All conditions follow the frozen protocols. Candidate selection and returned risks were audited against the raw arrays.
Each table entry contains 96 searches in 12 task/source-bank contexts. Confidence intervals resample contexts; they are descriptive with only three source banks.

## Robustness

| Condition | Method | Success | Mean MSE | Bank success percentages |
|---|---|---:|---:|---|
| early_128/e0 | VAE + release | 40/96 (41.7%) | 0.291571 | 37.5, 46.9, 40.6 |
| early_128/e0 | Components + release | 48/96 (50.0%) | 0.249901 | 43.8, 53.1, 53.1 |
| early_128/e0 | Components, hard tying | 48/96 (50.0%) | 0.249946 | 43.8, 53.1, 53.1 |
| early_128/e0 | Replay + release | 0/96 (0.0%) | 0.499718 | 0.0, 0.0, 0.0 |
| early_128/e0 | Uniform + release | 0/96 (0.0%) | 0.500312 | 0.0, 0.0, 0.0 |
| early_128/e0 | Oracle + release | 96/96 (100.0%) | 0 | 100.0, 100.0, 100.0 |
| early_128/e0 | Dense SGD | 0/96 (0.0%) | 0.50059 | 0.0, 0.0, 0.0 |
| early_128/e0 | Dense Adam | 1/96 (1.0%) | 0.495231 | 0.0, 3.1, 0.0 |
| early_32/e0 | VAE + release | 40/96 (41.7%) | 0.291571 | 37.5, 46.9, 40.6 |
| early_32/e0 | Components + release | 48/96 (50.0%) | 0.249901 | 43.8, 53.1, 53.1 |
| early_32/e0 | Components, hard tying | 48/96 (50.0%) | 0.249946 | 43.8, 53.1, 53.1 |
| early_32/e0 | Replay + release | 0/96 (0.0%) | 0.499718 | 0.0, 0.0, 0.0 |
| early_32/e0 | Uniform + release | 0/96 (0.0%) | 0.500312 | 0.0, 0.0, 0.0 |
| early_32/e0 | Oracle + release | 96/96 (100.0%) | 0 | 100.0, 100.0, 100.0 |
| early_32/e0 | Dense SGD | 0/96 (0.0%) | 0.50059 | 0.0, 0.0, 0.0 |
| early_32/e0 | Dense Adam | 1/96 (1.0%) | 0.495231 | 0.0, 3.1, 0.0 |
| early_8/e0 | VAE + release | 4/96 (4.2%) | 0.478718 | 3.1, 0.0, 9.4 |
| early_8/e0 | Components + release | 0/96 (0.0%) | 0.500075 | 0.0, 0.0, 0.0 |
| early_8/e0 | Components, hard tying | 0/96 (0.0%) | 0.50025 | 0.0, 0.0, 0.0 |
| early_8/e0 | Replay + release | 0/96 (0.0%) | 0.5002 | 0.0, 0.0, 0.0 |
| early_8/e0 | Uniform + release | 0/96 (0.0%) | 0.500264 | 0.0, 0.0, 0.0 |
| early_8/e0 | Oracle + release | 96/96 (100.0%) | 0 | 100.0, 100.0, 100.0 |
| early_8/e0 | Dense SGD | 0/96 (0.0%) | 0.50059 | 0.0, 0.0, 0.0 |
| early_8/e0 | Dense Adam | 1/96 (1.0%) | 0.495231 | 0.0, 3.1, 0.0 |
| noise_0.25/e0 | VAE + release | 40/96 (41.7%) | 0.291571 | 37.5, 46.9, 40.6 |
| noise_0.25/e0 | Components + release | 48/96 (50.0%) | 0.249901 | 43.8, 53.1, 53.1 |
| noise_0.25/e0 | Components, hard tying | 48/96 (50.0%) | 0.249946 | 43.8, 53.1, 53.1 |
| noise_0.25/e0 | Replay + release | 0/96 (0.0%) | 0.499718 | 0.0, 0.0, 0.0 |
| noise_0.25/e0 | Uniform + release | 0/96 (0.0%) | 0.500312 | 0.0, 0.0, 0.0 |
| noise_0.25/e0 | Oracle + release | 96/96 (100.0%) | 0 | 100.0, 100.0, 100.0 |
| noise_0.25/e0 | Dense SGD | 0/96 (0.0%) | 0.50059 | 0.0, 0.0, 0.0 |
| noise_0.25/e0 | Dense Adam | 1/96 (1.0%) | 0.495231 | 0.0, 3.1, 0.0 |
| noise_0.5/e0 | VAE + release | 44/96 (45.8%) | 0.270747 | 28.1, 65.6, 43.8 |
| noise_0.5/e0 | Components + release | 16/96 (16.7%) | 0.41616 | 12.5, 25.0, 12.5 |
| noise_0.5/e0 | Components, hard tying | 1/96 (1.0%) | 0.489166 | 0.0, 3.1, 0.0 |
| noise_0.5/e0 | Replay + release | 0/96 (0.0%) | 0.49951 | 0.0, 0.0, 0.0 |
| noise_0.5/e0 | Uniform + release | 0/96 (0.0%) | 0.500306 | 0.0, 0.0, 0.0 |
| noise_0.5/e0 | Oracle + release | 96/96 (100.0%) | 0 | 100.0, 100.0, 100.0 |
| noise_0.5/e0 | Dense SGD | 0/96 (0.0%) | 0.50059 | 0.0, 0.0, 0.0 |
| noise_0.5/e0 | Dense Adam | 1/96 (1.0%) | 0.495231 | 0.0, 3.1, 0.0 |
| noise_0.5/e0.3 | VAE + release | 41/96 (42.7%) | 0.282627 | 25.0, 65.6, 37.5 |
| noise_0.5/e0.3 | Components + release | 17/96 (17.7%) | 0.412509 | 12.5, 28.1, 12.5 |
| noise_0.5/e0.3 | Components, hard tying | 1/96 (1.0%) | 0.489748 | 0.0, 3.1, 0.0 |
| noise_0.5/e0.3 | Replay + release | 0/96 (0.0%) | 0.499479 | 0.0, 0.0, 0.0 |
| noise_0.5/e0.3 | Uniform + release | 0/96 (0.0%) | 0.50029 | 0.0, 0.0, 0.0 |
| noise_0.5/e0.3 | Oracle + release | 96/96 (100.0%) | 0 | 100.0, 100.0, 100.0 |
| noise_0.5/e0.3 | Dense SGD | 0/96 (0.0%) | 0.500465 | 0.0, 0.0, 0.0 |
| noise_0.5/e0.3 | Dense Adam | 1/96 (1.0%) | 0.495255 | 0.0, 3.1, 0.0 |
| noise_0/e0 | VAE + release | 40/96 (41.7%) | 0.291571 | 37.5, 46.9, 40.6 |
| noise_0/e0 | Components + release | 48/96 (50.0%) | 0.249901 | 43.8, 53.1, 53.1 |
| noise_0/e0 | Components, hard tying | 48/96 (50.0%) | 0.249946 | 43.8, 53.1, 53.1 |
| noise_0/e0 | Replay + release | 0/96 (0.0%) | 0.499718 | 0.0, 0.0, 0.0 |
| noise_0/e0 | Uniform + release | 0/96 (0.0%) | 0.500312 | 0.0, 0.0, 0.0 |
| noise_0/e0 | Oracle + release | 96/96 (100.0%) | 0 | 100.0, 100.0, 100.0 |
| noise_0/e0 | Dense SGD | 0/96 (0.0%) | 0.50059 | 0.0, 0.0, 0.0 |
| noise_0/e0 | Dense Adam | 1/96 (1.0%) | 0.495231 | 0.0, 3.1, 0.0 |
| noise_0/e0.1 | VAE + release | 40/96 (41.7%) | 0.291565 | 37.5, 46.9, 40.6 |
| noise_0/e0.1 | Components + release | 48/96 (50.0%) | 0.249917 | 43.8, 53.1, 53.1 |
| noise_0/e0.1 | Components, hard tying | 49/96 (51.0%) | 0.247293 | 43.8, 56.2, 53.1 |
| noise_0/e0.1 | Replay + release | 0/96 (0.0%) | 0.499709 | 0.0, 0.0, 0.0 |
| noise_0/e0.1 | Uniform + release | 0/96 (0.0%) | 0.500326 | 0.0, 0.0, 0.0 |
| noise_0/e0.1 | Oracle + release | 96/96 (100.0%) | 0 | 100.0, 100.0, 100.0 |
| noise_0/e0.1 | Dense SGD | 0/96 (0.0%) | 0.500556 | 0.0, 0.0, 0.0 |
| noise_0/e0.1 | Dense Adam | 1/96 (1.0%) | 0.495235 | 0.0, 3.1, 0.0 |
| noise_0/e0.3 | VAE + release | 41/96 (42.7%) | 0.286361 | 40.6, 46.9, 40.6 |
| noise_0/e0.3 | Components + release | 48/96 (50.0%) | 0.249899 | 40.6, 56.2, 53.1 |
| noise_0/e0.3 | Components, hard tying | 48/96 (50.0%) | 0.271504 | 40.6, 56.2, 53.1 |
| noise_0/e0.3 | Replay + release | 0/96 (0.0%) | 0.499722 | 0.0, 0.0, 0.0 |
| noise_0/e0.3 | Uniform + release | 0/96 (0.0%) | 0.500317 | 0.0, 0.0, 0.0 |
| noise_0/e0.3 | Oracle + release | 96/96 (100.0%) | 0 | 100.0, 100.0, 100.0 |
| noise_0/e0.3 | Dense SGD | 0/96 (0.0%) | 0.500465 | 0.0, 0.0, 0.0 |
| noise_0/e0.3 | Dense Adam | 1/96 (1.0%) | 0.495255 | 0.0, 3.1, 0.0 |
| noise_0/e0.6 | VAE + release | 39/96 (40.6%) | 0.296864 | 40.6, 43.8, 37.5 |
| noise_0/e0.6 | Components + release | 47/96 (49.0%) | 0.255164 | 43.8, 53.1, 50.0 |
| noise_0/e0.6 | Components, hard tying | 0/96 (0.0%) | 0.329298 | 0.0, 0.0, 0.0 |
| noise_0/e0.6 | Replay + release | 0/96 (0.0%) | 0.499852 | 0.0, 0.0, 0.0 |
| noise_0/e0.6 | Uniform + release | 0/96 (0.0%) | 0.500313 | 0.0, 0.0, 0.0 |
| noise_0/e0.6 | Oracle + release | 96/96 (100.0%) | 0 | 100.0, 100.0, 100.0 |
| noise_0/e0.6 | Dense SGD | 0/96 (0.0%) | 0.500487 | 0.0, 0.0, 0.0 |
| noise_0/e0.6 | Dense Adam | 1/96 (1.0%) | 0.495185 | 0.0, 3.1, 0.0 |
| noise_0/e1 | VAE + release | 37/96 (38.5%) | 0.307383 | 40.6, 40.6, 34.4 |
| noise_0/e1 | Components + release | 46/96 (47.9%) | 0.26046 | 43.8, 50.0, 50.0 |
| noise_0/e1 | Components, hard tying | 0/96 (0.0%) | 0.412118 | 0.0, 0.0, 0.0 |
| noise_0/e1 | Replay + release | 0/96 (0.0%) | 0.500051 | 0.0, 0.0, 0.0 |
| noise_0/e1 | Uniform + release | 0/96 (0.0%) | 0.500277 | 0.0, 0.0, 0.0 |
| noise_0/e1 | Oracle + release | 96/96 (100.0%) | 0 | 100.0, 100.0, 100.0 |
| noise_0/e1 | Dense SGD | 0/96 (0.0%) | 0.500638 | 0.0, 0.0, 0.0 |
| noise_0/e1 | Dense Adam | 0/96 (0.0%) | 0.500448 | 0.0, 0.0, 0.0 |
| noise_0/e2 | VAE + release | 8/96 (8.3%) | 0.434478 | 6.2, 9.4, 9.4 |
| noise_0/e2 | Components + release | 10/96 (10.4%) | 0.419962 | 12.5, 6.2, 12.5 |
| noise_0/e2 | Components, hard tying | 0/96 (0.0%) | 0.498313 | 0.0, 0.0, 0.0 |
| noise_0/e2 | Replay + release | 0/96 (0.0%) | 0.500247 | 0.0, 0.0, 0.0 |
| noise_0/e2 | Uniform + release | 0/96 (0.0%) | 0.500308 | 0.0, 0.0, 0.0 |
| noise_0/e2 | Oracle + release | 58/96 (60.4%) | 0.130738 | 62.5, 65.6, 53.1 |
| noise_0/e2 | Dense SGD | 0/96 (0.0%) | 0.500542 | 0.0, 0.0, 0.0 |
| noise_0/e2 | Dense Adam | 0/96 (0.0%) | 0.500457 | 0.0, 0.0, 0.0 |
| noise_1/e0 | VAE + release | 4/96 (4.2%) | 0.478713 | 3.1, 3.1, 6.2 |
| noise_1/e0 | Components + release | 0/96 (0.0%) | 0.500194 | 0.0, 0.0, 0.0 |
| noise_1/e0 | Components, hard tying | 0/96 (0.0%) | 0.500308 | 0.0, 0.0, 0.0 |
| noise_1/e0 | Replay + release | 0/96 (0.0%) | 0.500222 | 0.0, 0.0, 0.0 |
| noise_1/e0 | Uniform + release | 0/96 (0.0%) | 0.500253 | 0.0, 0.0, 0.0 |
| noise_1/e0 | Oracle + release | 96/96 (100.0%) | 0 | 100.0, 100.0, 100.0 |
| noise_1/e0 | Dense SGD | 0/96 (0.0%) | 0.50059 | 0.0, 0.0, 0.0 |
| noise_1/e0 | Dense Adam | 1/96 (1.0%) | 0.495231 | 0.0, 3.1, 0.0 |

## Two-layer tanh

| Condition | Method | Success | Mean MSE | Bank success percentages |
|---|---|---:|---:|---|
| two_layer_tanh | VAE | 72/96 (75.0%) | 0.0898782 | 75.0, 71.9, 78.1 |
| two_layer_tanh | Components | 78/96 (81.2%) | 0.0569095 | 84.4, 78.1, 81.2 |
| two_layer_tanh | Components + release | 80/96 (83.3%) | 0.00763945 | 90.6, 84.4, 75.0 |
| two_layer_tanh | Partition replay | 0/96 (0.0%) | 0.306869 | 0.0, 0.0, 0.0 |
| two_layer_tanh | Uniform | 0/96 (0.0%) | 0.488044 | 0.0, 0.0, 0.0 |
| two_layer_tanh | Oracle partition | 96/96 (100.0%) | 0.00167971 | 100.0, 100.0, 100.0 |
| two_layer_tanh | Dense SGD | 72/96 (75.0%) | 0.0103209 | 75.0, 75.0, 75.0 |
| two_layer_tanh | Dense Adam | 96/96 (100.0%) | 0.00047818 | 100.0, 100.0, 100.0 |
| two_layer_tanh | Numerical transfer + Adam | 96/96 (100.0%) | 0.000507878 | 100.0, 100.0, 100.0 |

## Source diagnostics

```json
[
  {
    "seed": 9101,
    "condition": "early_128",
    "source_steps": 128,
    "source_noise": 0.0,
    "original_sha256": "18d037ae3fa5c106211f5cc23cc79f679617df835b9d60a2ba75eee9cfae447a",
    "components": [
      [
        1,
        4,
        5,
        6,
        9,
        12,
        16,
        17,
        18,
        21,
        23,
        24,
        25,
        29,
        30,
        31
      ],
      [
        0,
        11,
        14,
        19
      ],
      [
        2,
        3,
        7,
        22
      ],
      [
        8,
        10,
        20,
        27
      ],
      [
        13,
        15,
        26,
        28
      ]
    ],
    "support": 16,
    "permutation_diagnostic": [
      18,
      30,
      5,
      3,
      17,
      14,
      7,
      2,
      26,
      16,
      28,
      20,
      15,
      12,
      21,
      11,
      1,
      6,
      22,
      19,
      29,
      8,
      4,
      24,
      0,
      9,
      10,
      27,
      13,
      23,
      31,
      25
    ],
    "source_mean_population": 5.122274160385132e-07,
    "assignment_bit_error_diagnostic": 0.0,
    "exact_assignment_fraction_diagnostic": 1.0
  },
  {
    "seed": 9101,
    "condition": "early_32",
    "source_steps": 32,
    "source_noise": 0.0,
    "original_sha256": "18d037ae3fa5c106211f5cc23cc79f679617df835b9d60a2ba75eee9cfae447a",
    "components": [
      [
        1,
        4,
        5,
        6,
        9,
        12,
        16,
        17,
        18,
        21,
        23,
        24,
        25,
        29,
        30,
        31
      ],
      [
        0,
        11,
        14,
        19
      ],
      [
        2,
        3,
        7,
        22
      ],
      [
        8,
        10,
        20,
        27
      ],
      [
        13,
        15,
        26,
        28
      ]
    ],
    "support": 16,
    "permutation_diagnostic": [
      18,
      30,
      5,
      3,
      17,
      14,
      7,
      2,
      26,
      16,
      28,
      20,
      15,
      12,
      21,
      11,
      1,
      6,
      22,
      19,
      29,
      8,
      4,
      24,
      0,
      9,
      10,
      27,
      13,
      23,
      31,
      25
    ],
    "source_mean_population": 0.00938006117939949,
    "assignment_bit_error_diagnostic": 0.0,
    "exact_assignment_fraction_diagnostic": 1.0
  },
  {
    "seed": 9101,
    "condition": "early_8",
    "source_steps": 8,
    "source_noise": 0.0,
    "original_sha256": "18d037ae3fa5c106211f5cc23cc79f679617df835b9d60a2ba75eee9cfae447a",
    "components": [
      [
        12
      ],
      [
        0
      ],
      [
        1
      ],
      [
        2
      ],
      [
        3
      ],
      [
        4
      ],
      [
        5
      ],
      [
        6
      ],
      [
        7
      ],
      [
        8
      ],
      [
        9
      ],
      [
        10
      ],
      [
        11
      ],
      [
        13
      ],
      [
        14
      ],
      [
        15
      ],
      [
        16
      ],
      [
        17
      ],
      [
        18
      ],
      [
        19
      ],
      [
        20
      ],
      [
        21
      ],
      [
        22
      ],
      [
        23
      ],
      [
        24
      ],
      [
        25
      ],
      [
        26
      ],
      [
        27
      ],
      [
        28
      ],
      [
        29
      ],
      [
        30
      ],
      [
        31
      ]
    ],
    "support": 2147483648,
    "permutation_diagnostic": [
      18,
      30,
      5,
      3,
      17,
      14,
      7,
      2,
      26,
      16,
      28,
      20,
      15,
      12,
      21,
      11,
      1,
      6,
      22,
      19,
      29,
      8,
      4,
      24,
      0,
      9,
      10,
      27,
      13,
      23,
      31,
      25
    ],
    "source_mean_population": 0.2400490641593933,
    "assignment_bit_error_diagnostic": 0.14453125,
    "exact_assignment_fraction_diagnostic": 0.0
  },
  {
    "seed": 9101,
    "condition": "noise_0",
    "source_steps": null,
    "source_noise": 0.0,
    "original_sha256": "18d037ae3fa5c106211f5cc23cc79f679617df835b9d60a2ba75eee9cfae447a",
    "components": [
      [
        1,
        4,
        5,
        6,
        9,
        12,
        16,
        17,
        18,
        21,
        23,
        24,
        25,
        29,
        30,
        31
      ],
      [
        0,
        11,
        14,
        19
      ],
      [
        2,
        3,
        7,
        22
      ],
      [
        8,
        10,
        20,
        27
      ],
      [
        13,
        15,
        26,
        28
      ]
    ],
    "support": 16,
    "permutation_diagnostic": [
      18,
      30,
      5,
      3,
      17,
      14,
      7,
      2,
      26,
      16,
      28,
      20,
      15,
      12,
      21,
      11,
      1,
      6,
      22,
      19,
      29,
      8,
      4,
      24,
      0,
      9,
      10,
      27,
      13,
      23,
      31,
      25
    ],
    "source_mean_population": 1.862645149230957e-09,
    "assignment_bit_error_diagnostic": 0.0,
    "exact_assignment_fraction_diagnostic": 1.0
  },
  {
    "seed": 9101,
    "condition": "noise_0.25",
    "source_steps": null,
    "source_noise": 0.25,
    "original_sha256": "18d037ae3fa5c106211f5cc23cc79f679617df835b9d60a2ba75eee9cfae447a",
    "components": [
      [
        1,
        4,
        5,
        6,
        9,
        12,
        16,
        17,
        18,
        21,
        23,
        24,
        25,
        29,
        30,
        31
      ],
      [
        0,
        11,
        14,
        19
      ],
      [
        2,
        3,
        7,
        22
      ],
      [
        8,
        10,
        20,
        27
      ],
      [
        13,
        15,
        26,
        28
      ]
    ],
    "support": 16,
    "permutation_diagnostic": [
      18,
      30,
      5,
      3,
      17,
      14,
      7,
      2,
      26,
      16,
      28,
      20,
      15,
      12,
      21,
      11,
      1,
      6,
      22,
      19,
      29,
      8,
      4,
      24,
      0,
      9,
      10,
      27,
      13,
      23,
      31,
      25
    ],
    "source_mean_population": 1.862645149230957e-09,
    "assignment_bit_error_diagnostic": 0.0,
    "exact_assignment_fraction_diagnostic": 1.0
  },
  {
    "seed": 9101,
    "condition": "noise_0.5",
    "source_steps": null,
    "source_noise": 0.5,
    "original_sha256": "18d037ae3fa5c106211f5cc23cc79f679617df835b9d60a2ba75eee9cfae447a",
    "components": [
      [
        5,
        6,
        9,
        12,
        17,
        18,
        21,
        23,
        24,
        25,
        29,
        30,
        31
      ],
      [
        0,
        11,
        14,
        19
      ],
      [
        1
      ],
      [
        2
      ],
      [
        3
      ],
      [
        4
      ],
      [
        7
      ],
      [
        8,
        10,
        20
      ],
      [
        13,
        15,
        26
      ],
      [
        16
      ],
      [
        22
      ],
      [
        27
      ],
      [
        28
      ]
    ],
    "support": 4096,
    "permutation_diagnostic": [
      18,
      30,
      5,
      3,
      17,
      14,
      7,
      2,
      26,
      16,
      28,
      20,
      15,
      12,
      21,
      11,
      1,
      6,
      22,
      19,
      29,
      8,
      4,
      24,
      0,
      9,
      10,
      27,
      13,
      23,
      31,
      25
    ],
    "source_mean_population": 1.862645149230957e-09,
    "assignment_bit_error_diagnostic": 0.015625,
    "exact_assignment_fraction_diagnostic": 0.5625
  },
  {
    "seed": 9101,
    "condition": "noise_1",
    "source_steps": null,
    "source_noise": 1.0,
    "original_sha256": "18d037ae3fa5c106211f5cc23cc79f679617df835b9d60a2ba75eee9cfae447a",
    "components": [
      [
        0
      ],
      [
        1
      ],
      [
        2
      ],
      [
        3
      ],
      [
        4
      ],
      [
        5,
        30
      ],
      [
        6
      ],
      [
        7
      ],
      [
        8
      ],
      [
        9
      ],
      [
        10
      ],
      [
        11
      ],
      [
        12
      ],
      [
        13
      ],
      [
        14
      ],
      [
        15
      ],
      [
        16
      ],
      [
        17
      ],
      [
        18
      ],
      [
        19
      ],
      [
        20
      ],
      [
        21
      ],
      [
        22
      ],
      [
        23
      ],
      [
        24
      ],
      [
        25
      ],
      [
        26
      ],
      [
        27
      ],
      [
        28
      ],
      [
        29
      ],
      [
        31
      ]
    ],
    "support": 2147483648,
    "permutation_diagnostic": [
      18,
      30,
      5,
      3,
      17,
      14,
      7,
      2,
      26,
      16,
      28,
      20,
      15,
      12,
      21,
      11,
      1,
      6,
      22,
      19,
      29,
      8,
      4,
      24,
      0,
      9,
      10,
      27,
      13,
      23,
      31,
      25
    ],
    "source_mean_population": 1.862645149230957e-09,
    "assignment_bit_error_diagnostic": 0.1484375,
    "exact_assignment_fraction_diagnostic": 0.0
  },
  {
    "seed": 9102,
    "condition": "early_128",
    "source_steps": 128,
    "source_noise": 0.0,
    "original_sha256": "d7d3a3b69e11dd6a5e8ef1e5deebc0557562bcf38abfb5c7025898f1ab025298",
    "components": [
      [
        7,
        8,
        9,
        11,
        12,
        13,
        14,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        28,
        30
      ],
      [
        0,
        15,
        29,
        31
      ],
      [
        1,
        4,
        26,
        27
      ],
      [
        2,
        5,
        10,
        17
      ],
      [
        3,
        6,
        16,
        25
      ]
    ],
    "support": 16,
    "permutation_diagnostic": [
      27,
      11,
      21,
      3,
      12,
      19,
      2,
      14,
      15,
      31,
      20,
      6,
      23,
      9,
      16,
      29,
      4,
      18,
      1,
      17,
      8,
      25,
      7,
      22,
      0,
      5,
      13,
      10,
      24,
      28,
      30,
      26
    ],
    "source_mean_population": 9.685754776000977e-08,
    "assignment_bit_error_diagnostic": 0.0,
    "exact_assignment_fraction_diagnostic": 1.0
  },
  {
    "seed": 9102,
    "condition": "early_32",
    "source_steps": 32,
    "source_noise": 0.0,
    "original_sha256": "d7d3a3b69e11dd6a5e8ef1e5deebc0557562bcf38abfb5c7025898f1ab025298",
    "components": [
      [
        7,
        8,
        9,
        11,
        12,
        13,
        14,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        28,
        30
      ],
      [
        0,
        15,
        29,
        31
      ],
      [
        1,
        4,
        26,
        27
      ],
      [
        2,
        5,
        10,
        17
      ],
      [
        3,
        6,
        16,
        25
      ]
    ],
    "support": 16,
    "permutation_diagnostic": [
      27,
      11,
      21,
      3,
      12,
      19,
      2,
      14,
      15,
      31,
      20,
      6,
      23,
      9,
      16,
      29,
      4,
      18,
      1,
      17,
      8,
      25,
      7,
      22,
      0,
      5,
      13,
      10,
      24,
      28,
      30,
      26
    ],
    "source_mean_population": 0.009051423519849777,
    "assignment_bit_error_diagnostic": 0.0,
    "exact_assignment_fraction_diagnostic": 1.0
  },
  {
    "seed": 9102,
    "condition": "early_8",
    "source_steps": 8,
    "source_noise": 0.0,
    "original_sha256": "d7d3a3b69e11dd6a5e8ef1e5deebc0557562bcf38abfb5c7025898f1ab025298",
    "components": [
      [
        8,
        19
      ],
      [
        0
      ],
      [
        1
      ],
      [
        2
      ],
      [
        3
      ],
      [
        4
      ],
      [
        5
      ],
      [
        6
      ],
      [
        7
      ],
      [
        9
      ],
      [
        10
      ],
      [
        11
      ],
      [
        12
      ],
      [
        13
      ],
      [
        14
      ],
      [
        15
      ],
      [
        16
      ],
      [
        17
      ],
      [
        18
      ],
      [
        20,
        24
      ],
      [
        21
      ],
      [
        22
      ],
      [
        23
      ],
      [
        25
      ],
      [
        26
      ],
      [
        27
      ],
      [
        28
      ],
      [
        29
      ],
      [
        30
      ],
      [
        31
      ]
    ],
    "support": 536870912,
    "permutation_diagnostic": [
      27,
      11,
      21,
      3,
      12,
      19,
      2,
      14,
      15,
      31,
      20,
      6,
      23,
      9,
      16,
      29,
      4,
      18,
      1,
      17,
      8,
      25,
      7,
      22,
      0,
      5,
      13,
      10,
      24,
      28,
      30,
      26
    ],
    "source_mean_population": 0.25572073459625244,
    "assignment_bit_error_diagnostic": 0.181640625,
    "exact_assignment_fraction_diagnostic": 0.0
  },
  {
    "seed": 9102,
    "condition": "noise_0",
    "source_steps": null,
    "source_noise": 0.0,
    "original_sha256": "d7d3a3b69e11dd6a5e8ef1e5deebc0557562bcf38abfb5c7025898f1ab025298",
    "components": [
      [
        7,
        8,
        9,
        11,
        12,
        13,
        14,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        28,
        30
      ],
      [
        0,
        15,
        29,
        31
      ],
      [
        1,
        4,
        26,
        27
      ],
      [
        2,
        5,
        10,
        17
      ],
      [
        3,
        6,
        16,
        25
      ]
    ],
    "support": 16,
    "permutation_diagnostic": [
      27,
      11,
      21,
      3,
      12,
      19,
      2,
      14,
      15,
      31,
      20,
      6,
      23,
      9,
      16,
      29,
      4,
      18,
      1,
      17,
      8,
      25,
      7,
      22,
      0,
      5,
      13,
      10,
      24,
      28,
      30,
      26
    ],
    "source_mean_population": 1.6763806343078613e-08,
    "assignment_bit_error_diagnostic": 0.0,
    "exact_assignment_fraction_diagnostic": 1.0
  },
  {
    "seed": 9102,
    "condition": "noise_0.25",
    "source_steps": null,
    "source_noise": 0.25,
    "original_sha256": "d7d3a3b69e11dd6a5e8ef1e5deebc0557562bcf38abfb5c7025898f1ab025298",
    "components": [
      [
        7,
        8,
        9,
        11,
        12,
        13,
        14,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        28,
        30
      ],
      [
        0,
        15,
        29,
        31
      ],
      [
        1,
        4,
        26,
        27
      ],
      [
        2,
        5,
        10,
        17
      ],
      [
        3,
        6,
        16,
        25
      ]
    ],
    "support": 16,
    "permutation_diagnostic": [
      27,
      11,
      21,
      3,
      12,
      19,
      2,
      14,
      15,
      31,
      20,
      6,
      23,
      9,
      16,
      29,
      4,
      18,
      1,
      17,
      8,
      25,
      7,
      22,
      0,
      5,
      13,
      10,
      24,
      28,
      30,
      26
    ],
    "source_mean_population": 1.6763806343078613e-08,
    "assignment_bit_error_diagnostic": 0.0,
    "exact_assignment_fraction_diagnostic": 1.0
  },
  {
    "seed": 9102,
    "condition": "noise_0.5",
    "source_steps": null,
    "source_noise": 0.5,
    "original_sha256": "d7d3a3b69e11dd6a5e8ef1e5deebc0557562bcf38abfb5c7025898f1ab025298",
    "components": [
      [
        7,
        11,
        12,
        14,
        18,
        22,
        23,
        28
      ],
      [
        0
      ],
      [
        1,
        4,
        26
      ],
      [
        2,
        5,
        10,
        17
      ],
      [
        3,
        6,
        16,
        25
      ],
      [
        8
      ],
      [
        9
      ],
      [
        13,
        20
      ],
      [
        15,
        29,
        31
      ],
      [
        19
      ],
      [
        21
      ],
      [
        24
      ],
      [
        27
      ],
      [
        30
      ]
    ],
    "support": 8192,
    "permutation_diagnostic": [
      27,
      11,
      21,
      3,
      12,
      19,
      2,
      14,
      15,
      31,
      20,
      6,
      23,
      9,
      16,
      29,
      4,
      18,
      1,
      17,
      8,
      25,
      7,
      22,
      0,
      5,
      13,
      10,
      24,
      28,
      30,
      26
    ],
    "source_mean_population": 1.6763806343078613e-08,
    "assignment_bit_error_diagnostic": 0.03515625,
    "exact_assignment_fraction_diagnostic": 0.375
  },
  {
    "seed": 9102,
    "condition": "noise_1",
    "source_steps": null,
    "source_noise": 1.0,
    "original_sha256": "d7d3a3b69e11dd6a5e8ef1e5deebc0557562bcf38abfb5c7025898f1ab025298",
    "components": [
      [
        22
      ],
      [
        0
      ],
      [
        1
      ],
      [
        2
      ],
      [
        3
      ],
      [
        4
      ],
      [
        5
      ],
      [
        6
      ],
      [
        7
      ],
      [
        8
      ],
      [
        9
      ],
      [
        10
      ],
      [
        11
      ],
      [
        12
      ],
      [
        13
      ],
      [
        14
      ],
      [
        15
      ],
      [
        16
      ],
      [
        17
      ],
      [
        18
      ],
      [
        19
      ],
      [
        20
      ],
      [
        21
      ],
      [
        23
      ],
      [
        24
      ],
      [
        25
      ],
      [
        26
      ],
      [
        27
      ],
      [
        28
      ],
      [
        29
      ],
      [
        30
      ],
      [
        31
      ]
    ],
    "support": 2147483648,
    "permutation_diagnostic": [
      27,
      11,
      21,
      3,
      12,
      19,
      2,
      14,
      15,
      31,
      20,
      6,
      23,
      9,
      16,
      29,
      4,
      18,
      1,
      17,
      8,
      25,
      7,
      22,
      0,
      5,
      13,
      10,
      24,
      28,
      30,
      26
    ],
    "source_mean_population": 1.6763806343078613e-08,
    "assignment_bit_error_diagnostic": 0.169921875,
    "exact_assignment_fraction_diagnostic": 0.0
  },
  {
    "seed": 9103,
    "condition": "early_128",
    "source_steps": 128,
    "source_noise": 0.0,
    "original_sha256": "c16b89865c434d00fb3c6a3c623f37a03e236200e49d78d18046ae0574c2b578",
    "components": [
      [
        0,
        1,
        2,
        3,
        5,
        6,
        9,
        12,
        14,
        17,
        20,
        22,
        24,
        26,
        29,
        30
      ],
      [
        4,
        13,
        21,
        28
      ],
      [
        7,
        8,
        10,
        16
      ],
      [
        11,
        15,
        25,
        27
      ],
      [
        18,
        19,
        23,
        31
      ]
    ],
    "support": 16,
    "permutation_diagnostic": [
      24,
      14,
      16,
      22,
      5,
      7,
      17,
      10,
      11,
      25,
      12,
      19,
      9,
      2,
      23,
      18,
      13,
      0,
      27,
      29,
      6,
      4,
      30,
      28,
      8,
      20,
      1,
      21,
      3,
      15,
      31,
      26
    ],
    "source_mean_population": 2.250075340270996e-06,
    "assignment_bit_error_diagnostic": 0.0,
    "exact_assignment_fraction_diagnostic": 1.0
  },
  {
    "seed": 9103,
    "condition": "early_32",
    "source_steps": 32,
    "source_noise": 0.0,
    "original_sha256": "c16b89865c434d00fb3c6a3c623f37a03e236200e49d78d18046ae0574c2b578",
    "components": [
      [
        0,
        1,
        2,
        3,
        5,
        6,
        9,
        12,
        14,
        17,
        20,
        22,
        24,
        26,
        29,
        30
      ],
      [
        4,
        13,
        21,
        28
      ],
      [
        7,
        8,
        10,
        16
      ],
      [
        11,
        15,
        25,
        27
      ],
      [
        18,
        19,
        23,
        31
      ]
    ],
    "support": 16,
    "permutation_diagnostic": [
      24,
      14,
      16,
      22,
      5,
      7,
      17,
      10,
      11,
      25,
      12,
      19,
      9,
      2,
      23,
      18,
      13,
      0,
      27,
      29,
      6,
      4,
      30,
      28,
      8,
      20,
      1,
      21,
      3,
      15,
      31,
      26
    ],
    "source_mean_population": 0.010581573471426964,
    "assignment_bit_error_diagnostic": 0.0,
    "exact_assignment_fraction_diagnostic": 1.0
  },
  {
    "seed": 9103,
    "condition": "early_8",
    "source_steps": 8,
    "source_noise": 0.0,
    "original_sha256": "c16b89865c434d00fb3c6a3c623f37a03e236200e49d78d18046ae0574c2b578",
    "components": [
      [
        24
      ],
      [
        0
      ],
      [
        1
      ],
      [
        2
      ],
      [
        3
      ],
      [
        4
      ],
      [
        5
      ],
      [
        6
      ],
      [
        7
      ],
      [
        8
      ],
      [
        9
      ],
      [
        10
      ],
      [
        11
      ],
      [
        12
      ],
      [
        13
      ],
      [
        14
      ],
      [
        15
      ],
      [
        16
      ],
      [
        17
      ],
      [
        18
      ],
      [
        19
      ],
      [
        20
      ],
      [
        21
      ],
      [
        22
      ],
      [
        23
      ],
      [
        25
      ],
      [
        26
      ],
      [
        27
      ],
      [
        28
      ],
      [
        29
      ],
      [
        30
      ],
      [
        31
      ]
    ],
    "support": 2147483648,
    "permutation_diagnostic": [
      24,
      14,
      16,
      22,
      5,
      7,
      17,
      10,
      11,
      25,
      12,
      19,
      9,
      2,
      23,
      18,
      13,
      0,
      27,
      29,
      6,
      4,
      30,
      28,
      8,
      20,
      1,
      21,
      3,
      15,
      31,
      26
    ],
    "source_mean_population": 0.26277434825897217,
    "assignment_bit_error_diagnostic": 0.166015625,
    "exact_assignment_fraction_diagnostic": 0.0
  },
  {
    "seed": 9103,
    "condition": "noise_0",
    "source_steps": null,
    "source_noise": 0.0,
    "original_sha256": "c16b89865c434d00fb3c6a3c623f37a03e236200e49d78d18046ae0574c2b578",
    "components": [
      [
        0,
        1,
        2,
        3,
        5,
        6,
        9,
        12,
        14,
        17,
        20,
        22,
        24,
        26,
        29,
        30
      ],
      [
        4,
        13,
        21,
        28
      ],
      [
        7,
        8,
        10,
        16
      ],
      [
        11,
        15,
        25,
        27
      ],
      [
        18,
        19,
        23,
        31
      ]
    ],
    "support": 16,
    "permutation_diagnostic": [
      24,
      14,
      16,
      22,
      5,
      7,
      17,
      10,
      11,
      25,
      12,
      19,
      9,
      2,
      23,
      18,
      13,
      0,
      27,
      29,
      6,
      4,
      30,
      28,
      8,
      20,
      1,
      21,
      3,
      15,
      31,
      26
    ],
    "source_mean_population": 7.450580596923828e-09,
    "assignment_bit_error_diagnostic": 0.0,
    "exact_assignment_fraction_diagnostic": 1.0
  },
  {
    "seed": 9103,
    "condition": "noise_0.25",
    "source_steps": null,
    "source_noise": 0.25,
    "original_sha256": "c16b89865c434d00fb3c6a3c623f37a03e236200e49d78d18046ae0574c2b578",
    "components": [
      [
        0,
        1,
        2,
        3,
        5,
        6,
        9,
        12,
        14,
        17,
        20,
        22,
        24,
        26,
        29,
        30
      ],
      [
        4,
        13,
        21,
        28
      ],
      [
        7,
        8,
        10,
        16
      ],
      [
        11,
        15,
        25,
        27
      ],
      [
        18,
        19,
        23,
        31
      ]
    ],
    "support": 16,
    "permutation_diagnostic": [
      24,
      14,
      16,
      22,
      5,
      7,
      17,
      10,
      11,
      25,
      12,
      19,
      9,
      2,
      23,
      18,
      13,
      0,
      27,
      29,
      6,
      4,
      30,
      28,
      8,
      20,
      1,
      21,
      3,
      15,
      31,
      26
    ],
    "source_mean_population": 7.450580596923828e-09,
    "assignment_bit_error_diagnostic": 0.0,
    "exact_assignment_fraction_diagnostic": 1.0
  },
  {
    "seed": 9103,
    "condition": "noise_0.5",
    "source_steps": null,
    "source_noise": 0.5,
    "original_sha256": "c16b89865c434d00fb3c6a3c623f37a03e236200e49d78d18046ae0574c2b578",
    "components": [
      [
        0,
        3,
        6,
        9,
        12,
        17,
        20,
        22,
        24,
        26,
        29
      ],
      [
        1
      ],
      [
        2
      ],
      [
        4
      ],
      [
        5
      ],
      [
        7
      ],
      [
        8
      ],
      [
        10
      ],
      [
        11,
        15,
        27
      ],
      [
        13,
        21,
        28
      ],
      [
        14
      ],
      [
        16
      ],
      [
        18
      ],
      [
        19
      ],
      [
        23
      ],
      [
        25
      ],
      [
        30
      ],
      [
        31
      ]
    ],
    "support": 131072,
    "permutation_diagnostic": [
      24,
      14,
      16,
      22,
      5,
      7,
      17,
      10,
      11,
      25,
      12,
      19,
      9,
      2,
      23,
      18,
      13,
      0,
      27,
      29,
      6,
      4,
      30,
      28,
      8,
      20,
      1,
      21,
      3,
      15,
      31,
      26
    ],
    "source_mean_population": 7.450580596923828e-09,
    "assignment_bit_error_diagnostic": 0.03125,
    "exact_assignment_fraction_diagnostic": 0.3125
  },
  {
    "seed": 9103,
    "condition": "noise_1",
    "source_steps": null,
    "source_noise": 1.0,
    "original_sha256": "c16b89865c434d00fb3c6a3c623f37a03e236200e49d78d18046ae0574c2b578",
    "components": [
      [
        0
      ],
      [
        1
      ],
      [
        2
      ],
      [
        3
      ],
      [
        4
      ],
      [
        5
      ],
      [
        6
      ],
      [
        7
      ],
      [
        8
      ],
      [
        9
      ],
      [
        10
      ],
      [
        11
      ],
      [
        12
      ],
      [
        13
      ],
      [
        14
      ],
      [
        15
      ],
      [
        16
      ],
      [
        17
      ],
      [
        18
      ],
      [
        19
      ],
      [
        20
      ],
      [
        21
      ],
      [
        22
      ],
      [
        23
      ],
      [
        24
      ],
      [
        25
      ],
      [
        26
      ],
      [
        27
      ],
      [
        28
      ],
      [
        29
      ],
      [
        30
      ],
      [
        31
      ]
    ],
    "support": 4294967296,
    "permutation_diagnostic": [
      24,
      14,
      16,
      22,
      5,
      7,
      17,
      10,
      11,
      25,
      12,
      19,
      9,
      2,
      23,
      18,
      13,
      0,
      27,
      29,
      6,
      4,
      30,
      28,
      8,
      20,
      1,
      21,
      3,
      15,
      31,
      26
    ],
    "source_mean_population": 7.450580596923828e-09,
    "assignment_bit_error_diagnostic": 0.16796875,
    "exact_assignment_fraction_diagnostic": 0.0
  },
  {
    "bank": "bank_s10101",
    "seconds": 9.002713736146688,
    "source_tasks": 16,
    "fits": 64,
    "gradient_example_presentations": 16777216,
    "mean_evaluation_mse": 0.0009484768379479647,
    "exact_partition_fraction_diagnostic": 1.0,
    "prior": {
      "components": [
        [
          0,
          1,
          2,
          4,
          5,
          6,
          7,
          9,
          10,
          14,
          15,
          18,
          22,
          27,
          28,
          31
        ],
        [
          3,
          11,
          16,
          21
        ],
        [
          8,
          12,
          19,
          30
        ],
        [
          13,
          23,
          26,
          29
        ],
        [
          17,
          20,
          24,
          25
        ]
      ],
      "support": 16,
      "permutation_diagnostic": [
        22,
        6,
        17,
        26,
        24,
        14,
        8,
        16,
        5,
        25,
        30,
        27,
        4,
        19,
        23,
        31,
        28,
        13,
        15,
        3,
        10,
        29,
        0,
        20,
        12,
        11,
        18,
        7,
        1,
        21,
        2,
        9
      ]
    }
  },
  {
    "bank": "bank_s10102",
    "seconds": 8.966704055666924,
    "source_tasks": 16,
    "fits": 64,
    "gradient_example_presentations": 16777216,
    "mean_evaluation_mse": 0.0009148968965746462,
    "exact_partition_fraction_diagnostic": 1.0,
    "prior": {
      "components": [
        [
          0,
          4,
          5,
          6,
          12,
          14,
          15,
          17,
          18,
          19,
          20,
          23,
          24,
          26,
          29,
          31
        ],
        [
          1,
          7,
          9,
          10
        ],
        [
          2,
          22,
          28,
          30
        ],
        [
          3,
          8,
          11,
          27
        ],
        [
          13,
          16,
          21,
          25
        ]
      ],
      "support": 16,
      "permutation_diagnostic": [
        9,
        21,
        5,
        13,
        25,
        22,
        16,
        18,
        12,
        19,
        20,
        10,
        15,
        28,
        31,
        23,
        26,
        1,
        17,
        7,
        8,
        27,
        4,
        6,
        30,
        29,
        14,
        11,
        2,
        0,
        3,
        24
      ]
    }
  },
  {
    "bank": "bank_s10103",
    "seconds": 8.78380329720676,
    "source_tasks": 16,
    "fits": 64,
    "gradient_example_presentations": 16777216,
    "mean_evaluation_mse": 0.0005955686792731285,
    "exact_partition_fraction_diagnostic": 1.0,
    "prior": {
      "components": [
        [
          0,
          3,
          5,
          6,
          7,
          8,
          9,
          12,
          13,
          14,
          15,
          17,
          18,
          24,
          25,
          30
        ],
        [
          1,
          19,
          21,
          27
        ],
        [
          2,
          11,
          26,
          29
        ],
        [
          4,
          10,
          22,
          31
        ],
        [
          16,
          20,
          23,
          28
        ]
      ],
      "support": 16,
      "permutation_diagnostic": [
        22,
        2,
        11,
        8,
        21,
        6,
        16,
        25,
        15,
        0,
        20,
        12,
        24,
        1,
        31,
        7,
        29,
        30,
        14,
        5,
        28,
        4,
        19,
        27,
        17,
        9,
        13,
        3,
        26,
        10,
        23,
        18
      ]
    }
  }
]
```

The full JSON summary retains context-bootstrap intervals, continuous losses, per-bank rates, and failure counts. Raw selected parameters and all candidate errors are in the exported run directories.
