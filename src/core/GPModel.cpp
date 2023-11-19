#include <iostream>
#include "GPModel.h"
#include "GradientDescent.h"
#include <Eigen/Dense>  // Eigen library for matrix operations
#include <functional>
using namespace std;

GPModel::GPModel(KernelBase* kernel) : kernel(kernel) {}

void GPModel::fit(const Eigen::MatrixXd& x_train, const Eigen::VectorXd& y_train, double sigma) {

    this->x_train = x_train;
    this->y_train = y_train;
    this->sigma = sigma;
    
    // Define parameters for gradient descent
    const Eigen::VectorXd start_point = Eigen::VectorXd::Constant(1., 1.);
    double lr = 0.01;
    int iters = 1000;
    const Eigen::VectorXd lower_bounds = Eigen::VectorXd::Constant(1e-5, 1e-5);
    const Eigen::VectorXd upper_bounds = Eigen::VectorXd::Constant(1e3, 1e3);


    auto bound_log_min_likelihood = std::bind(&GPModel::log_min_likelihood, this, std::placeholders::_1);
    GradientDescentOptimizer optimizer(bound_log_min_likelihood, start_point, lr, iters, lower_bounds, upper_bounds);
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

    kernel->setHyperparameters(params);
    computeKernelInverse();
    long int n_train = static_cast<int>(x_train.rows());
    double log_likelihood = -0.5*y_train.transpose()*K_inv*y_train - 0.5*log(K_inv.determinant()) - n_train/2.0*log(2*M_PI);
    // Return negative log likelihood because we are using a minimizer but we want to maximize the likelihood
    return -log_likelihood;
}

void GPModel::computeKernelInverse() {
    // Compute the inverse of the kernel matrix
    kernel->computeMatrix(x_train, K_inv);
    K_inv = K_inv.inverse();
}
