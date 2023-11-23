#ifndef GPMODEL_H
#define GPMODEL_H

#include "KernelBase.h"
#include "GradientDescent.h"
#include <Eigen/Dense>

class GPModel {
public:
    // Constructor & Destructor
    GPModel(KernelBase* kernel, double sigma);
    ~GPModel() = default;

    // Fit the Gaussian Process to training data
    void fit(const Eigen::MatrixXd& x_train, const Eigen::VectorXd& y_train);

    // Predict the output and its standard debiation for given input(s) x
    Eigen::MatrixXd predict(const Eigen::MatrixXd& x_test);


    // Set the kernel for the Gaussian Process
    void setKernel(KernelBase* kernel);

private:
    KernelBase* kernel;
    Eigen::MatrixXd x_train;
    Eigen::VectorXd y_train;
    Eigen::MatrixXd K_inv;  // Inverse of kernel matrix
    double sigma;

    void computeKernelInverse();
    double log_min_likelihood(const Eigen::VectorXd& params);
};

#endif // GPMODEL_H
