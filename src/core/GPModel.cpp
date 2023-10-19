#include <iostream>
#include "GPModel.h"
#include <Eigen/Dense>  // Eigen library for matrix operations
using namespace std;

GPModel::GPModel(KernelBase* kernel) : kernel(kernel) {}

void GPModel::fit(const Eigen::MatrixXd& x_train, const Eigen::VectorXd& y_train, double sigma) {

    this->x_train = x_train;
    this->y_train = y_train;
    this->sigma = sigma;
 
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

void GPModel::computeKernelInverse() {
    cout << "debug 3.1" << endl;
    kernel->computeMatrix(x_train, K_inv);
    cout << "debug 3.2" << endl;
    long int n_train = static_cast<int>(x_train.rows());
    K_inv = K_inv + sigma*sigma*Eigen::MatrixXd::Identity(n_train, n_train);
    K_inv = K_inv.inverse();
    cout << "debug 3.3" << endl;
}
