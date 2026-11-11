when trying with a pretrained resnet18 network -> validation f1, overfitting on the data
the RandomFlip with prob=0.5 has no effect, i tend to believe that the transformations are irrelevant
in these case, because mostly the tumors are central and surrounded by patches of skin

## Transformations ideas:
### Useful transformations:
from my perspective the RandomCrop metehod might yield good results if the cropping happens from the center of the photo, it is an idea worth trying.
SpatialCrop / CenterSpatialCrop / RandomSpatialCrop
ResizeWithPadAndCrop -> to make sure that most of the central part of the tumar is covered (first resize to 512 * 512) -> afterward central crop

Solution combine: Resize [512, 512] -> CenterSpatialCrop [288, 288]-> RandomSpatialCrop [224, 224] (for train functions)
(reduces the image size too much down to 19% of the initial entry)

### Useless transformations:
padding -> the photos are already pretty big and they are resized to smaller sizes what
would be the point in doing this.


## Oversampling + Undersampling idea:
initially the difference between the highest and the lowest class was -> 50 times (between the NV and DF classess) really bad
reduced the number of NV classes to 40% of their initial entries
doubled the number of entries for AK, DF, VASC, SCC classes now the difference is around 10 times (still using cross entropy loss)


## Citations:
### Citation 1:
Ahmad, I., Khubaib, M., & Altowaijri, S. M. (2026). Benchmarking Class Imbalance Mitigation Strategies Across Deep CNN Architectures for Skin Cancer Classification. Diagnostics, 16(16), 2571. https://doi.org/10.3390/diagnostics16162571
#### Info found:
random oversampling, random undersampling -> already in use.
Adaptive syntethic sampling, SMOTE-LOF ?? -> multiple sampling strategies are available from there

-- did not tried FocalLoss yet
curriculum learning, Cummulative Learning, class balanced optimization
