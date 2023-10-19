#include "RBFKernel.h"
#include <cmath>  // For exp and pow functions

RBFKernel::RBFKernel(double length_scale) : length_scale(length_scale) {}

double RBFKernel::compute(double x1, double x2) const {
    double distance = x1 - x2;
    return exp(-pow(distance, 2) / (2.0 * pow(length_scale, 2)));
}

double RBFKernel::getLengthScale() const {
    return length_scale;
}

void RBFKernel::setLengthScale(double length_scale) {
    this->length_scale = length_scale;
}
