#ifndef KERNELBASE_H
#define KERNELBASE_H

#include <Eigen/Dense>

class KernelBase {
public:
    KernelBase() = default;
    virtual ~KernelBase() = default;

    // Virtual function to compute the covariance between two points
    virtual double compute(double x1, double x2) const = 0;

    // Virtual function to compute the kernel matrix for a set of points
    virtual void computeMatrix(const Eigen::VectorXd& X, Eigen::MatrixXd& K) const;
};

#endif // KERNELBASE_H
