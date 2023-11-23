#include <iostream>
#include "core/GPModel.h"
#include "optimizers/GradientDescent.h"
#include <Eigen/Dense>  // Eigen library for matrix operations
#include <functional>
#include <numbers>
using namespace std;

GPModel::GPModel(KernelBase* kernel, double sigma) : kernel(kernel), sigma(sigma) {}

void GPModel::fit(const Eigen::MatrixXd& x_train, const Eigen::VectorXd& y_train) {

    this->x_train = x_train;
    this->y_train = y_train;
    
    // Define parameters for gradient descent
    double lr = 0.01;
    int iters = 1000;
    int n_params = kernel->getHyperparameters().size() + 1; // +1 for sigma
    const Eigen::VectorXd start_point = Eigen::VectorXd::Constant(n_params, 1.);
    const Eigen::VectorXd lower_bounds = Eigen::VectorXd::Constant(n_params, 1e-5);
    const Eigen::VectorXd upper_bounds = Eigen::VectorXd::Constant(n_params, 1e3);


    auto bound_log_min_likelihood = std::bind(&GPModel::log_min_likelihood, this, std::placeholders::_1);

    GradientDescentOptimizer optimizer(
        bound_log_min_likelihood,
        start_point,
        lr,
        iters,
        lower_bounds,
        upper_bounds);

    Eigen::VectorXd params = optimizer.minimize();
    kernel->setHyperparameters(params);

    computeKernelInverse();

}

Eigen::MatrixXd GPModel::predict(const Eigen::MatrixXd& x_test) {
    long int n_train = static_cast<int>(x_train.rows());
    long int n_test = static_cast<int>(x_test.rows());

    Eigen::VectorXd predictions(n_test);
    Eigen::VectorXd std(n_test);

    for (long int i=0; i < n_test; i++) {
        Eigen::VectorXd k_star(n_train);
        for (long int j=0; j < n_train; j++) {
            k_star(j) = kernel->compute(x_test.row(i), x_train.row(j));
        }
        // Calculate the mean prediction
        double prediction = k_star.dot(K_inv*y_train);
        predictions(i) = prediction;

        // Calculate the standard deviation

        double k_star_star = kernel->compute(x_test.row(i), x_test.row(i));  
        double variance = k_star_star - k_star.dot(K_inv*k_star);
        std(i) = sqrt(variance) + sigma;
    }

    Eigen::MatrixXd predictionsWithStd(n_test,2);
    predictionsWithStd << predictions, std;

    cout << predictionsWithStd.rows()<<" " << predictionsWithStd.cols() << endl;

    return predictionsWithStd;
}

void GPModel::setKernel(KernelBase* kernel) {
    this->kernel = kernel;
}

double GPModel::log_min_likelihood(const Eigen::VectorXd& params) {

    sigma = params(0);
    kernel->setHyperparameters(params.tail(params.size()-1));
    computeKernelInverse();
    long int n_train = static_cast<int>(x_train.rows());
    Eigen::MatrixXd K_tilde = K_inv + sigma*sigma*Eigen::MatrixXd::Identity(n_train, n_train);
    double log_likelihood = -0.5*y_train.transpose()*(K_tilde)*y_train - 0.5*log(K_tilde.determinant()) - n_train/2.0*log(2*std::numbers::pi);
    // Return negative log likelihood because we are using a minimizer but we want to maximize the likelihood
    return -log_likelihood;
}

void GPModel::computeKernelInverse() {
    // Compute the inverse of the kernel matrix
    kernel->computeMatrix(x_train, K_inv);
    K_inv = K_inv.inverse();
}
