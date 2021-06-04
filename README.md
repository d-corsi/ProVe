# ProVe (UAI'21)
A Python implementation of ProVe. ProVe is formal verifier for safety property for Artificial Neural Netowrk, presentend at the 37th conference on Uncertainty in Artificial Intelligence. You can find detailed descriptions of ProVe in the conference paper.
This repository contains the implementations of ProVe and a set of models and properties to reproduce the results inside the paper.

## Abstract
In the last years, neural networks achieved groundbreaking successes in a wide variety of applications. However, for safety critical tasks, such as robotics and healthcare, it is necessary to provide some specific guarantees before the deployment in a real world context. Even in these scenarios, where high cost equipment and human safety are involved, the evaluation of the models is usually performed with the standard metrics (i.e., cumulative reward or success rate).
In this paper, we introduce a novel metric for the evaluation of models in safety critical tasks, the violation rate. We build our work upon the concept of formal verification for neural networks, providing a new formulation for the safety properties that aims to ensure that the agent always makes rational decisions. To perform this evaluation, we present ProVe (Property Verifier), a novel approach based on the interval algebra, designed for the analysis of our novel behavioral properties. 
We apply our method to different domains (i.e., mapless navigation for mobile robots, trajectory generation for manipulators, and the standard ACAS benchmark). Results show that the violation rate computed by ProVe provides a good evaluation for the safety of trained models.


## Prerequisite and Installation
To run the tool you need to install the following python packages:

- Numpy
- [CuPy](https://docs.cupy.dev/en/stable/install.html#install-cupy) (we refer to the official page for a complete tutorial)
- Tensorflow

## Running ACAS Experiments
To run the deafault ACAS experiments described inside the Paper, run the following command:
```
python main.py -ACAS *property*
```
where *property* is an integer between [1] and [15], corresponding to the desired property. For a comlete description of the ACAS properties we refer to the [Neurify](https://arxiv.org/abs/1809.08098) [Wang et al.] paper.

## Running Water Navigation Experiments
To run the additional experiments on the Water Narvagation problem, run the following command:
```
python main.py -water 
```

## Custom properties
To run a custom property, please modify the configuration file at 'config/example.yaml', copy your keras model inside the folder 'trained_model/' and run the command:
```
python main.py -custom
```

## Authors
*  **Davide Corsi** - davide.corsi@univr.it
*  **Enrico Marchesini** - enrico.marchesini@univr.it
*  **Alessandro Farinelli** - alessandro.farinelli@univr.it