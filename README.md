## An interpretable data fusion model for Visual Field and Optical Coherence Tomography measurements from glaucoma patients.
This repository is a part of the paper "Using fused data from perimetry and optical coherence tomography to improve the detection of visual field progression in glaucoma".

![AE_architecture](https://github.com/lcapacitor/glaucoma-vf-oct-data-fusion/blob/main/figures/ae_architecture.jpg)

The overall architecture of the autoencoder (AE) data fusion model. The input to the model is a vector that includes pointwise differential light sensitivity thresholds from visual field (VF) testing (52-dimensional vector), retinal nerve fiber layer (RNFL) thickness profile (256-dimensional vector), and patient’s age at the time of the test (scalar). The encoder network, constructed with a two-hidden layer multilayer perceptron (MLP) model, processes the input vector and generates a 52-dimensional encoding vector as the AE-fused data. The decoder network, a symmetrically structured MLP model, aims to reconstruct the input data from the encoding vector. The reconstruction loss ($L_{rec}$) is the mean squared error (MSE) between the input and output vectors of the AE data fusion model. The encoding loss ($L_{enc}$) is the MSE between the AE-fused data and the measured VF. The training objective is to minimize the convex combination of the reconstruction loss and the encoding loss, weighted by a scalar $\lambda$. 

## Authors
Yan(Leo) Li<sup>1</sup>, Moshe Eizenman<sup>2,3</sup>, Runjie B. Shi<sup>3,4</sup>, Yvonne M. Buys<sup>2</sup>, Graham E. Trope<sup>2</sup>, Willy Wong<sup>1,4</sup>
* <sup>1</sup> The Edward S. Rogers Sr. Department of Electrical & Computer Engineering, University of Toronto
* <sup>2</sup> Department of Ophthalmology & Vision Sciences, University of Toronto
* <sup>3</sup> Temerty Faculty of Medicine, University of Toronto
* <sup>4</sup> Institute of Biomedical Engineering, University of Toronto

## Usage
#### 1. Train the AE data fusion model with mock data and 10-fold cross-validation:
```
python cv_train.py --num_cv 10
```
> [!NOTE]
> See all training arguments in the cv_train.py file.

#### 2. Evalute trained AE data fusion models with mock data:
```
python cv_evaluate.py
```

#### 3. Show data fusion results for a glaucoma patient with mild visual field defects.
```
python data_fusion_examples.py --eye mild
```
![mild_eye](https://github.com/lcapacitor/glaucoma-vf-oct-data-fusion/blob/main/figures/example_mild.jpg)
For this eye, the RNFL thickness measurements (the rightmost plot) exhibit notable thinning in the 225° to 315° region. As such, the mean RNFL thickness (70.3 µm) falls below the normal range of 75.0 µm to 107.2 µm suggested by the Cirrus HD-OCT device. This localized RNFL thinning is reflected in the AE-fused data as a VF defect in more depression in the superior nasal region of the field (the middle VF plot). It should be noted that the region where the VF loss lies in the AE-fused data matches the area of RNFL thinning according to the Garway-Heath structure-function map. Since the VF and OCT data describe the same defect, the AE-fused data tend to have a lower MD (-3.1 dB in the right VF plot) than the MD of measured VF data (-1.5 dB in the left VF plot). 


#### 4. Show data fusion results for a glaucoma patient with moderate visual field defects.
```
python data_fusion_examples.py --eye moderate
```
![moderate_eye](https://github.com/lcapacitor/glaucoma-vf-oct-data-fusion/blob/main/figures/example_moderate.jpg)
The defect pattern of the measured VF (the left VF plot) is a superposition of an actual VF loss in the superior field and lens rim artifacts. Considering that the RNFL thickness in this eye is overall normal (mRNLFT= 88.9 µm), the impact of lens rim artifacts is removed in the AE-fused data (the middle VF plot), leading to a milder VF loss (MD= -5.7 dB), while maintaining the arcuate defect pattern in the superior field. Note that the reconstructed VF data (the right VF plot) maintains good consistency with the measured VF data in terms of the shape and the depth of the defect (MD= -7.8 dB), showing that the information from the measured VF test has been embedded into the AE-fused data. 


#### 5. Show data fusion results for a glaucoma patient with severe visual field defects.
```
python data_fusion_examples.py --eye severe
```
![severe_eye](https://github.com/lcapacitor/glaucoma-vf-oct-data-fusion/blob/main/figures/example_severe.jpg)
In this advanced glaucoma case, the floor effect dominates the RNFL thickness measurements, plateauing at the level of around 50 µm (the rightmost plot). As a result, the AE-fused data (the middle VF plot) is more dependent on data from VF testing and, correspondingly, shows greater consistency (MD= -13.6 dB) with the measured VF data (MD= -13.8 dB).

## License
MIT
