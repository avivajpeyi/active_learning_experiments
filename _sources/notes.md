# Active learning notes

## Papers on Active learning 
### Methods/Algos

1. uncertainty-based sampling: least confident ([Lewis and Catlett](https://www.sciencedirect.com/science/article/pii/B978155860335650026X?via%3Dihub)), max margin and max entropy
1. committee-based algorithms: vote entropy, consensus entropy and max disagreement ([Cohn et al.](http://www.cs.northwestern.edu/~pardo/courses/mmml/papers/active_learning/improving_generalization_with_active_learning_ML94.pdf))
1. multilabel strategies: SVM binary minimum ([Brinker](https://link.springer.com/chapter/10.1007%2F3-540-31314-1_24)), max loss, mean max loss, ([Li et al.](http://dx.doi.org/10.1109/ICIP.2004.1421535)) MinConfidence, MeanConfidence, MinScore, MeanScore ([Esuli and Sebastiani](http://dx.doi.org/10.1007/978-3-642-00958-7_12))
1. expected error reduction: binary and log loss ([Roy and McCallum](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.588.5666&rep=rep1&type=pdf))
1. Bayesian optimization: probability of improvement, expected improvement and upper confidence bound ([Snoek et al.](https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf))
1. batch active learning: ranked batch-mode sampling ([Cardoso et al.](https://www.sciencedirect.com/science/article/pii/S0020025516313949))
1. information density framework ([McCallum and Nigam](http://www.kamalnigam.com/papers/emactive-icml98.pdf))
1. stream-based sampling ([Atlas et al.](https://papers.nips.cc/paper/261-training-connectionist-networks-with-queries-and-selective-sampling.pdf))
1. active regression with max standard deviance sampling for Gaussian processes or ensemble regressors


### How to estimate uncertainty in Deep Learning networks

* [Excellent tutorial from AGW on Bayesian Deep Learning](https://icml.cc/virtual/2020/tutorial/5750)

* This tut is inspired by his publication [Bayesian Deep Learning and a Probabilistic Perspective of Generalization](https://arxiv.org/abs/2002.08791)

* [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/pdf/1506.02142.pdf) (Gal and Ghahramani, 2016) This describes Monte-Carlo Dropout, a way to estimate uncertainty through stochastic dropout at test time


* [Bayesian Uncertainty Estimation for Batch Normalized Deep Networks](https://arxiv.org/abs/1802.06455) (Teye et al. 2018) This describes Monte-Carlo BatchNorm, a way to estimate uncertainty through random batch norm parameters at test time


* [Bayesian Deep Learning and a Probabilistic Perspective of Generalization](https://arxiv.org/abs/2002.08791) (Gordon Wilson and Izmailov, 2020) Presentation of multi-SWAG a mix between VI and Ensembles.


* [Advances in Variational inference](https://arxiv.org/pdf/1711.05597.pdf) (Zhang et al, 2018)
Gives a quick introduction to VI and the most recent advances.

* [A Simple Baseline for Bayesian Uncertainty in Deep Learning](https://arxiv.org/abs/1902.02476) (Maddox et al. 2019)
Presents SWAG, an easy way to create ensembles.

### Bayesian active learning

* [Deep Bayesian Active Learning with Image Data](https://arxiv.org/pdf/1703.02910.pdf) (Gal and Islam and Ghahramani, 2017) Fundamental paper on how to do Bayesian active learning.

* [Sampling bias in active learning](http://cseweb.ucsd.edu/~dasgupta/papers/twoface.pdf) (Dasgupta 2009) Presents sampling bias and how to solve it by combining heuristics and random selection.

* [Bayesian Active Learning for Classification and Preference Learning](https://arxiv.org/pdf/1112.5745.pdf) (Houlsby et al. 2011) Fundamental paper on one of the main heuristic BALD.


### GP and active learning

- [Exploration of lattice Hamiltonians for functional and structural discovery via Gaussian process-based exploration–exploitation](https://pubs.aip.org/aip/jap/article/128/16/164304/568362/Exploration-of-lattice-Hamiltonians-for-functional)
![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*FwG-cE5ABw_KMUrkJ_o2vQ.jpeg)

- [Active Learning for Deep Gaussian Process Surrogates](https://www.tandfonline.com/doi/full/10.1080/00401706.2021.2008505)

- [Actively learning GP dynamics](https://arxiv.org/abs/1911.09946)


### General
- Review paper: https://arxiv.org/abs/2009.00236
- Book on Active learning: https://www.manning.com/books/human-in-the-loop-machine-learning
- "Towards Robust Deep Active Learning for Scientific Computing" https://arxiv.org/abs/2201.12632
- Deep bayesian active learning + image https://arxiv.org/abs/1703.02910
- A Comparative Survey of Deep Active Learning: https://arxiv.org/abs/2203.13450


## Tutorials/worksops

* [Google's Active Learning Playground](https://github.com/google/active-learning): This is a python module for experimenting with different active learning algorithms.
* [deep-active-learning](https://github.com/ej0cl6/deep-active-learning): Python implementations of the following active learning algorithms
* [PyTorch Active Learning](https://github.com/rmunro/pytorch_active_learning): Library for common Active Learning methods
* [active-learning-workshop](https://github.com/Azure/active-learning-workshop):KDD 2018 Hands-on Tutorial: Active learning and transfer learning at scale with R and Python. [PDF](https://github.com/Azure/active-learning-workshop/blob/master/active_learning_workshop.pdf)


## Codebases

1. Modal 

Built on scipy+sklearn
https://github.com/modAL-python/modAL


2. gpax

Gaussian processes + active learning! 
Very new, uses JAX ❤️ + numpyro
Problem: unstable/

https://github.com/ziatdinovmax/gpax/



3. Scikit-activeml
https://www.preprints.org/manuscript/202103.0194/v1
https://pypi.org/project/scikit-activeml/

Also built on scipy+sklearn

3. Baal

A " Bayesian active learning library" 
Built on pytorch 
Seems like focus on images
https://baal.readthedocs.io/en/latest/notebooks/compatibility/sklearn_tutorial/
https://baal.readthedocs.io/en/latest/
https://arxiv.org/abs/2006.09916


4. DeepAL
https://github.com/ej0cl6/deep-active-learning

5. adaptive

Adaptive sampling technique for 'learning' functional representation of the data
https://adaptive.readthedocs.io/en/latest/index.html
![](https://adaptive.readthedocs.io/en/latest/_static/logo_docs.webm)
https://github.com/python-adaptive/adaptive/tree/v1.0.0

6. AliPy
Agnostic of pytorch/sklearn/tflow 
https://github.com/NUAA-AL/alipy

7. libact
Features the [active learning by learning](http://www.csie.ntu.edu.tw/~htlin/paper/doc/aaai15albl.pdf) meta-strategy that allows the machine to automatically learn the best strategy on the fly. 

Works with sklearn

Hasnt been updated in 2 years
https://github.com/ntucllab/libact


8. botorch
A bayesian optimization framework built on pytorch
https://botorch.org/

