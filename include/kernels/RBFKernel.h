#ifndef RBFKERNEL_H
#define RBFKERNEL_H

#include "KernelBase.h"

class RBFKernel : public KernelBase {
public:
    // Constructor with the length-scale parameter
    RBFKernel(double length_scale, double sigma);
    
    // Override the compute method from KernelBase
    double compute(Eigen::VectorXd x1, Eigen::VectorXd x2) const override;

    // Getter and setter
    double getLengthScale() const;
    double getSigma() const;
    Eigen::VectorXd getHyperparameters() const override;
    void setHyperparameters(const Eigen::VectorXd& params) override;
    void setLengthScale(double length_scale);
    void setSigma(double sigma); 

private:
    double length_scale;
    double sigma;
};

#endif // RBFKERNEL_H
