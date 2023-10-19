#include <iostream>
#include "GPModel.h"
#include <Eigen/Dense>  // Eigen library for matrix operations
using namespace std;

GPModel::GPModel(KernelBase* kernel) : kernel(kernel) {}

void GPModel::fit(const Eigen::VectorXd& x_train, const Eigen::VectorXd& y_train) {

    this->x_train = x_train;
    this->y_train = y_train;
 
    computeKernelInverse();
}

Eigen::MatrixXd GPModel::predict(const Eigen::VectorXd& x_test) {
    size_t n_train = x_train.size();
    size_t n_test = x_test.size();

    Eigen::VectorXd predictions(n_test);
    Eigen::VectorXd std(n_test);

    for (size_t i=0; i < n_test; i++) {
        Eigen::VectorXd k_star(n_train);
        for (size_t j=0; j < n_train; j++) {
            k_star(j) = kernel->compute(x_test(i), x_train(j));
        }
        // Calculate the mean prediction
        double prediction = k_star.dot(K_inv*y_train);
        predictions[i] = prediction;

        // Calculate the standard deviation

        double k_star_star = kernel->compute(x_test(i), x_test(i));  
        double variance = k_star_star - k_star.dot(K_inv*k_star);
        std[i] = sqrt(variance);
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
    kernel->computeMatrix(x_train, K_inv);
    K_inv = K_inv.inverse();
}
