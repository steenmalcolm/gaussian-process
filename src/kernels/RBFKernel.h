#ifndef RBFKERNEL_H
#define RBFKERNEL_H

#include "KernelBase.h"

class RBFKernel : public KernelBase {
public:
    // Constructor with the length-scale parameter
    RBFKernel(double length_scale, double sigma);
    
    // Override the compute method from KernelBase
    double compute(Eigen::VectorXd x1, Eigen::VectorXd x2) const override;

    // Getter and setter for the length_scale parameter
    double getLengthScale() const;
    void setHyperparameters(const Eigen::VectorXd& params) override;
    void setLengthScale(double length_scale);

private:
    double length_scale;
};

#endif // RBFKERNEL_H
