#ifndef KERNELBASE_H
#define KERNELBASE_H

#include <Eigen/Dense>

class KernelBase {
public:
    KernelBase() = default;
    virtual ~KernelBase() = default;

    // Virtual function to compute the covariance between two points
    virtual double compute(Eigen::VectorXd x1, Eigen::VectorXd x2) const = 0;

    // Function to compute the kernel matrix for a set of points
    void computeMatrix(const Eigen::MatrixXd& X, Eigen::MatrixXd& K) const;

    // Virtual function to set the hyperparameters of the kernel
    virtual void setHyperparameters(const Eigen::VectorXd& params) = 0;

    // Getter and setter for the sigma parameter
    double getSigma() const;
    void setSigma(double sigma);

private:
    double sigma;
};

#endif // KERNELBASE_H
