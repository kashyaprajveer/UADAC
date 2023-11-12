# UADAC

Following program has been created to understand the effect of ensemble of Actor-critic's role in solving the same task. The intent behind this is while both actor-critic is seeing the same data however since they have a minute difference which is in terms of network initialization help in reducing the variance of the model. Thus it helps in improving the model's over all performance.

The dataset that is used is ml-100k, ML-1M and Jester. Currently ML-100k dataset is available in this repository.

Offline evalustion of the program has been shown in the experiments however in future online simulator will be used to further examine the effectiveness. The approach to reduce variance can be mathematically verified and has been a center of research in past few years.