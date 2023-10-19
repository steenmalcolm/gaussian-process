#ifndef RBFKERNEL_H
#define RBFKERNEL_H

#include "KernelBase.h"

class RBFKernel : public KernelBase {
public:
    // Constructor with the length-scale parameter
    RBFKernel(double length_scale);
    
    // Override the compute method from KernelBase
    double compute(double x1, double x2) const override;

    // Getter and setter for the length_scale parameter
    double getLengthScale() const;
    void setLengthScale(double length_scale);

private:
    double length_scale;
};

#endif // RBFKERNEL_H
