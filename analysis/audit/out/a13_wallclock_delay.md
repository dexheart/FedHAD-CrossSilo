# A13 - wall-clock / communication-delay audit (test3_comunicacao)

## alpha-0_5 folder: 1080 unique reports (0 byte-identical duplicates skipped)
hardware recorded: 1 x NVIDIA RTX A1000 / Intel(R) Core(TM) i7-14700: 1080
alpha: 0.5 | rounds: 10 | clients: 5

| method | dataset | delay | n | mean time (s) | sd |
|---|---|---|---|---|---|
| FedAVG | CIFAR10 | 0.05 | 30 | 387.63 | 2.73 |
| FedAVG | CIFAR10 | 0.1 | 30 | 388.29 | 2.45 |
| FedAVG | CIFAR10 | 0.5 | 30 | 399.58 | 2.66 |
| FedAVG | FashionMNIST | 0.05 | 30 | 131.54 | 11.32 |
| FedAVG | FashionMNIST | 0.1 | 30 | 133.46 | 10.15 |
| FedAVG | FashionMNIST | 0.5 | 30 | 145.75 | 11.40 |
| FedAVG | MNIST | 0.05 | 30 | 130.18 | 9.64 |
| FedAVG | MNIST | 0.1 | 30 | 132.91 | 10.63 |
| FedAVG | MNIST | 0.5 | 30 | 144.99 | 10.37 |
| FedAvgM | CIFAR10 | 0.05 | 30 | 389.02 | 1.90 |
| FedAvgM | CIFAR10 | 0.1 | 30 | 391.33 | 1.95 |
| FedAvgM | CIFAR10 | 0.5 | 30 | 402.91 | 2.63 |
| FedAvgM | FashionMNIST | 0.05 | 30 | 128.14 | 10.70 |
| FedAvgM | FashionMNIST | 0.1 | 30 | 130.24 | 11.21 |
| FedAvgM | FashionMNIST | 0.5 | 30 | 146.82 | 12.30 |
| FedAvgM | MNIST | 0.05 | 30 | 130.01 | 11.29 |
| FedAvgM | MNIST | 0.1 | 30 | 130.84 | 12.14 |
| FedAvgM | MNIST | 0.5 | 30 | 141.85 | 10.58 |
| FedHAD | CIFAR10 | 0.05 | 30 | 458.14 | 16.11 |
| FedHAD | CIFAR10 | 0.1 | 30 | 459.12 | 16.33 |
| FedHAD | CIFAR10 | 0.5 | 30 | 467.30 | 16.33 |
| FedHAD | FashionMNIST | 0.05 | 30 | 182.20 | 15.49 |
| FedHAD | FashionMNIST | 0.1 | 30 | 183.32 | 15.32 |
| FedHAD | FashionMNIST | 0.5 | 30 | 192.06 | 15.97 |
| FedHAD | MNIST | 0.05 | 30 | 183.04 | 15.19 |
| FedHAD | MNIST | 0.1 | 30 | 184.24 | 15.25 |
| FedHAD | MNIST | 0.5 | 30 | 192.87 | 15.28 |
| FedProx | CIFAR10 | 0.05 | 30 | 563.66 | 1.87 |
| FedProx | CIFAR10 | 0.1 | 30 | 566.21 | 1.56 |
| FedProx | CIFAR10 | 0.5 | 30 | 578.94 | 3.58 |
| FedProx | FashionMNIST | 0.05 | 30 | 215.93 | 16.16 |
| FedProx | FashionMNIST | 0.1 | 30 | 217.63 | 15.75 |
| FedProx | FashionMNIST | 0.5 | 30 | 229.89 | 15.88 |
| FedProx | MNIST | 0.05 | 30 | 217.03 | 15.34 |
| FedProx | MNIST | 0.1 | 30 | 218.61 | 15.10 |
| FedProx | MNIST | 0.5 | 30 | 230.95 | 14.97 |

## alpha-0_01 folder: 1080 unique reports (0 byte-identical duplicates skipped)
hardware recorded: 1 x NVIDIA A40 / AMD EPYC 9354P 32-Core Processor: 1080
alpha: 0.01 | rounds: 10 | clients: 5

| method | dataset | delay | n | mean time (s) | sd |
|---|---|---|---|---|---|
| FedAVG | CIFAR10 | 0.05 | 30 | 164.28 | 16.64 |
| FedAVG | CIFAR10 | 0.1 | 30 | 165.65 | 16.57 |
| FedAVG | CIFAR10 | 0.5 | 30 | 178.17 | 16.58 |
| FedAVG | FashionMNIST | 0.05 | 30 | 147.01 | 17.31 |
| FedAVG | FashionMNIST | 0.1 | 30 | 148.70 | 17.42 |
| FedAVG | FashionMNIST | 0.5 | 30 | 161.00 | 17.52 |
| FedAVG | MNIST | 0.05 | 30 | 147.18 | 17.14 |
| FedAVG | MNIST | 0.1 | 30 | 148.54 | 17.22 |
| FedAVG | MNIST | 0.5 | 30 | 161.15 | 17.29 |
| FedAvgM | CIFAR10 | 0.05 | 30 | 162.05 | 16.64 |
| FedAvgM | CIFAR10 | 0.1 | 30 | 163.54 | 16.45 |
| FedAvgM | CIFAR10 | 0.5 | 30 | 175.48 | 16.34 |
| FedAvgM | FashionMNIST | 0.05 | 30 | 144.75 | 17.31 |
| FedAvgM | FashionMNIST | 0.1 | 30 | 146.26 | 17.38 |
| FedAvgM | FashionMNIST | 0.5 | 30 | 158.06 | 17.28 |
| FedAvgM | MNIST | 0.05 | 30 | 144.64 | 16.90 |
| FedAvgM | MNIST | 0.1 | 30 | 146.11 | 17.13 |
| FedAvgM | MNIST | 0.5 | 30 | 158.16 | 16.98 |
| FedHAD | CIFAR10 | 0.05 | 30 | 176.84 | 26.04 |
| FedHAD | CIFAR10 | 0.1 | 30 | 178.07 | 26.23 |
| FedHAD | CIFAR10 | 0.5 | 30 | 186.41 | 26.05 |
| FedHAD | FashionMNIST | 0.05 | 30 | 146.75 | 23.55 |
| FedHAD | FashionMNIST | 0.1 | 30 | 147.66 | 23.30 |
| FedHAD | FashionMNIST | 0.5 | 30 | 156.19 | 23.56 |
| FedHAD | MNIST | 0.05 | 30 | 146.76 | 23.30 |
| FedHAD | MNIST | 0.1 | 30 | 147.63 | 23.33 |
| FedHAD | MNIST | 0.5 | 30 | 156.23 | 23.43 |
| FedProx | CIFAR10 | 0.05 | 30 | 226.14 | 23.55 |
| FedProx | CIFAR10 | 0.1 | 30 | 227.63 | 23.76 |
| FedProx | CIFAR10 | 0.5 | 30 | 240.02 | 23.74 |
| FedProx | FashionMNIST | 0.05 | 30 | 176.82 | 21.74 |
| FedProx | FashionMNIST | 0.1 | 30 | 178.56 | 21.97 |
| FedProx | FashionMNIST | 0.5 | 30 | 190.88 | 21.94 |
| FedProx | MNIST | 0.05 | 30 | 177.14 | 21.85 |
| FedProx | MNIST | 0.1 | 30 | 178.22 | 21.72 |
| FedProx | MNIST | 0.5 | 30 | 190.81 | 21.79 |

conflicting (non-identical) reports for the same cell: 0

## analise_resumo_test3_comunicacao.csv vs raw reports (max |diff| of the means, s)
  alpha-0_5 folder: 36 cells compared, max |diff| = 0.000 s, n mismatches = 0
  alpha-0_01 folder: 36 cells compared, max |diff| = 338.920 s, n mismatches = 0

## numbers quoted in the text (alpha-0_5 folder)
  FedHAD CIFAR10 0.05s: text 458.1 | raw mean 458.14 (n=30) | OK
  FedProx CIFAR10 0.05s: text 563.7 | raw mean 563.66 (n=30) | OK
  FedAVG CIFAR10 0.05s: text 387.6 | raw mean 387.63 (n=30) | OK
  increment 0.05 -> 0.5 s (text: 'between 9 and 19 s'): FedAVG/MNIST +14.8, FedAVG/FashionMNIST +14.2, FedAVG/CIFAR10 +12.0, FedAvgM/MNIST +11.8, FedAvgM/FashionMNIST +18.7, FedAvgM/CIFAR10 +13.9, FedProx/MNIST +13.9, FedProx/FashionMNIST +14.0, FedProx/CIFAR10 +15.3, FedHAD/MNIST +9.8, FedHAD/FashionMNIST +9.9, FedHAD/CIFAR10 +9.2 | min 9.2, max 18.7
  ordering FedProx > FedHAD > FedAvg, FedAvgM at every dataset and delay: True
