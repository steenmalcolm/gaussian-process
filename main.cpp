#include "KernelBase.h"
#include "RBFKernel.h"
#include "GPModel.h"
// #include "DataLoader.h"  // Assuming you have a DataLoader utility to load datasets.
#include <iostream>
#include <Eigen/Dense>
using namespace std;

int main() {
    // 1. Load dataset
    Eigen::VectorXd x_train(15), y_train(15);

    x_train << 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.,
               11., 12., 13., 14., 15.;
    y_train << 1., 4., 9., 16., 25., 36., 49., 64., 81., 100.,
               121., 144., 169., 196., 225.;

    // Assuming a DataLoader utility to load a hypothetical dataset from a CSV file.
    // DataLoader::loadFromCSV("data/sample_dataset.csv", x_train, y_train); 

    // 2. Instantiate an RBF kernel and set its length scale.
    double length_scale = 1.0;
    RBFKernel rbfKernel(length_scale);

    // 3. Train the Gaussian Process model on the dataset.
    GPModel gp(&rbfKernel);
    gp.fit(x_train, y_train);

    // 4. Predict on a new set of test points.
    Eigen::VectorXd x_test(3);
    x_test << 1.5, 2.5, 3.5;
    Eigen::MatrixXd predictionsWithStd(3, 2);
    predictionsWithStd = gp.predict(x_test);

    cout << predictionsWithStd << endl;

    // 5. Display the predictions.
    std::cout << "Predictions:" << std::endl;
    for (size_t i = 0; i < x_test.size(); ++i) {
        std::cout << "x = " << x_test[i] << ", y = " << predictionsWithStd(i,0)

          << "+-" << predictionsWithStd(i,1) << std::endl;
    }

    return 0;
}
