| commit   | model       | loss      | metric | description                                                                                                                   |
|----------|-------------|-----------|--------|-------------------------------------------------------------------------------------------------------------------------------|
| 4c6a4a2e | Trivial     | 55.145504 |        | Prediction is mean of target on processed train data                                                                          |
| 1d8472d4 | Baseline TS | 54.078785 |        | Pixel ts on S1 data only. optimizer: SGD(lr=0.1, momentum=0.9), scheduler: StepLR(step_size=6, gamma=0.5), n_epoch, bs=1 chip |
| 572494d5 | Baseline TS | 53.974094 |        | change batch_size: 1 -> 4                                                                                                     |
