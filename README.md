## An interpretable data fusion model for Visual Field and Optical Coherence Tomography measurements from glaucoma patients.
This repository is a part of the paper "Using fused data from perimetry and optical coherence tomography to improve the detection of visual field progression in glaucoma".

![AE_architecture](https://github.com/lcapacitor/glaucoma-vf-oct-data-fusion/blob/main/figures/ae_architecture.jpg)

The overall architecture of the autoencoder (AE) data fusion model. The input to the model is a vector that includes pointwise differential light sensitivity thresholds from visual field (VF) testing (52-dimensional vector), retinal nerve fiber layer (RNFL) thickness profile (256-dimensional vector), and patient’s age at the time of the test (scalar). The encoder network, constructed with a two-hidden layer multilayer perceptron (MLP) model, processes the input vector and generates a 52-dimensional encoding vector as the AE-fused data. The decoder network, a symmetrically structured MLP model, aims to reconstruct the input data from the encoding vector. The reconstruction loss ($L_{rec}$) is the mean squared error (MSE) between the input and output vectors of the AE data fusion model. The encoding loss ($L_{enc}$) is the MSE between the AE-fused data and the measured VF. The training objective is to minimize the convex combination of the reconstruction loss and the encoding loss, weighted by a scalar $\lambda$. 

### Authors
Yan(Leo) Li<sup>1</sup>, Moshe Eizenman<sup>2,3</sup>, Runjie B. Shi<sup>3,4</sup>, Yvonne M. Buys<sup>2</sup>, Graham E. Trope<sup>2</sup>, Willy Wong<sup>1,4</sup>
* <sup>1</sup> The Edward S. Rogers Sr. Department of Electrical & Computer Engineering, University of Toronto
* <sup>2</sup> Department of Ophthalmology & Vision Sciences, University of Toronto
* <sup>3</sup> Temerty Faculty of Medicine, University of Toronto
* <sup>4</sup> Institute of Biomedical Engineering, University of Toronto

### Usage
#### Train the AE data fusion model with mock data and 10-fold cross-validation:
```
python cv_train.py --num_cv 10
```

#### Evalute trained AE data fusion models with mock data:
```
python cv_evaluate.py
```

#### Show data fusion results for three eyes from the real data with mild, moderate, and severe visual field defects, respectively.
```
python data_fusion_examples.py --eye mild
```
![mild_eye](https://github.com/lcapacitor/glaucoma-vf-oct-data-fusion/blob/main/figures/example_mild.jpg)

```
python data_fusion_examples.py --eye moderate
```
![mild_eye](https://github.com/lcapacitor/glaucoma-vf-oct-data-fusion/blob/main/figures/example_moderate.jpg)

```
python data_fusion_examples.py --eye severe
```
![mild_eye](https://github.com/lcapacitor/glaucoma-vf-oct-data-fusion/blob/main/figures/example_severe.jpg)


## License
MIT
