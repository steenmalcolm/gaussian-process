#include <iostream>
#include "kernels/KernelBase.h"
#include <Eigen/Dense>
using namespace std;

void KernelBase::computeMatrix(const Eigen::MatrixXd & X, Eigen::MatrixXd & K) const {
    // This method fills the kernel matrix K based on the dataset X using the specific kernel function

    // Ensure the kernel matrix has the right dimensions
    double n = static_cast<double>(X.rows());
    K.conservativeResize(n,n);

    // Compute the kernel values
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; ++j) {
            K(i,j) = compute(X.row(i), X.row(j));
        }
    }
}
