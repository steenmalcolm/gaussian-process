#include "kernels/RBFKernel.h"
#include <cmath>

RBFKernel::RBFKernel(double length_scale, double sigma) : length_scale(length_scale), sigma(sigma) {}

double RBFKernel::compute(Eigen::VectorXd x1, Eigen::VectorXd x2) const {
    double distance = (x1 - x2).norm();
    return sigma*sigma*exp(-pow(distance, 2) / (2.0 * pow(length_scale, 2)));
}

void RBFKernel::setHyperparameters(const Eigen::VectorXd& params) {
    this->setSigma(params(0));
    this->setLengthScale(params(1));
}

double RBFKernel::getLengthScale() const {
    return length_scale;
}

double RBFKernel::getSigma() const {
    return sigma;
}

Eigen::VectorXd RBFKernel::getHyperparameters() const {
    Eigen::VectorXd params(2);
    params << sigma, length_scale;
    return params;
}

void RBFKernel::setLengthScale(double length_scale) {
    this->length_scale = length_scale;
}

void RBFKernel::setSigma(double sigma) {
    this->sigma = sigma;
}
