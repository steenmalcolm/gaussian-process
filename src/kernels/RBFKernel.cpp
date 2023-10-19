#include "RBFKernel.h"
#include <cmath>  // exp, pow

RBFKernel::RBFKernel(double length_scale) : length_scale(length_scale) {}

double RBFKernel::compute(Eigen::VectorXd x1, Eigen::VectorXd x2) const {
    double distance = (x1 - x2).norm();
    return exp(-pow(distance, 2) / (2.0 * pow(length_scale, 2)));
}

double RBFKernel::getLengthScale() const {
    return length_scale;
}

void RBFKernel::setLengthScale(double length_scale) {
    this->length_scale = length_scale;
}
