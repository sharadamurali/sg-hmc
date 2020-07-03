# Stochastic Gradient Hamiltonian Monte Carlo

A simple python implementation of Stochastic Gradient Hamiltonian Monte Carlo[1] for an application with synthetic data, described in [2]. 

The two parameters to be estimated, &theta;<sub>1</sub> and &theta;<sub>2</sub> are assumed to be drawn from normal distributions. 100 data points are generated, and the SGLD algorithm is run for 10,000 epochs. The history of the estimated parameters forms the posterior distribution, displayed below with the variation in the estimated gradient for the two parameters.

<img src="https://github.com/sharadamurali/sg-hmc/blob/master/images/sghmc_estimated_posterior_3.png?raw=true" alt="alt text" width="400"> <img src="https://github.com/sharadamurali/sg-hmc/blob/master/images/sghmc_grad.png?raw=true" alt="alt text" width="400">


## References
[1] Tianqi Chen, Emily Fox, and Carlos Guestrin. Stochastic gradient hamiltonian monte carlo. In Proceedings of the 31st International Conference on Machine Learning, volume 32 of Proceedings of Machine Learning Research, pages 1683–1691, 2014.

[2] Max Welling and Yee Whye Teh. Bayesian learning via stochastic gradient langevin dynamics. In Proceedings of the 28th International Conference on International Conference on Machine Learning, ICML’11, pages 681–688, 2011.
