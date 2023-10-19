#ifndef GPMODEL_H
#define GPMODEL_H

#include <Eigen/Dense>
#include "KernelBase.h"

class GPModel {
public:
    // Constructor & Destructor
    GPModel(KernelBase* kernel);
    ~GPModel() = default;

    // Fit the Gaussian Process to training data
    void fit(const Eigen::VectorXd& x_train, const Eigen::VectorXd& y_train, double sigma);

    // Predict the output and its standard debiation for given input(s) x
    Eigen::MatrixXd predict(const Eigen::VectorXd& x_test);


    // Set the kernel for the Gaussian Process
    void setKernel(KernelBase* kernel);

private:
    KernelBase* kernel;
    Eigen::VectorXd x_train;
    Eigen::VectorXd y_train;
    Eigen::MatrixXd K_inv;  // Inverse of kernel matrix
    double sigma;

    void computeKernelInverse();
};

#endif // GPMODEL_H
