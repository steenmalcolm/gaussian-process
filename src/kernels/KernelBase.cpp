#include <iostream>
#include "KernelBase.h"
#include <Eigen/Dense>
using namespace std;

void KernelBase::computeMatrix(const Eigen::VectorXd & X, Eigen::MatrixXd & K) const {
    // This method fills the kernel matrix K based on the dataset X using the specific kernel function

    // Ensure the kernel matrix has the right dimensions
    size_t n = X.size();
    K.conservativeResize(n,n);

    // Compute the kernel values
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; ++j) {
            K(i,j) = compute(X(i), X(j));
        }
    }
}
