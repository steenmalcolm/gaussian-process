#include "KernelBase.h"
#include "RBFKernel.h"
#include "GPModel.h"
// #include "DataLoader.h"  // Assuming you have a DataLoader utility to load datasets.
#include <iostream>
#include <Eigen/Dense>
#include <cmath>
using namespace std;

int main() {
    // 1. Load dataset
    Eigen::MatrixXd x_train(15*15,2);
    Eigen::VectorXd x(15), y_train(15*15);

    x << 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.,
               11., 12., 13., 14., 15.;
    for (int i=0; i<15; i++){
        for (int j=0; j<15; j++){
            x_train(15*i+j,0) = x(i);
            x_train(15*i+j,1) = x(j);
            // y_train is gaussian bell of x1 and x2
            y_train(15*i+j) = 10*exp(-pow(x(i)-3,2)/4.0) * exp(-pow(x(j)-3,2)/2.0);
        }
    }

    cout << "debug 1" << endl;
    // Assuming a DataLoader utility to load a hypothetical dataset from a CSV file.
    // DataLoader::loadFromCSV("data/sample_dataset.csv", x_train, y_train); 

    // 2. Instantiate an RBF kernel and set its length scale.
    double length_scale = 1.0;
    double sigma = 0.1;
    RBFKernel rbfKernel(length_scale);

    cout << "debug 2" << endl;
    // 3. Train the Gaussian Process model on the dataset.
    GPModel gp(&rbfKernel);
    cout << "debug 3" << endl;
    gp.fit(x_train, y_train, sigma);
    cout << "debug 4" << endl;

    // 4. Predict on a new set of test points.
    Eigen::MatrixXd x_test(15,2);
    x_test << 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,
              8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 
              13, 13, 14, 14, 15, 15;
    Eigen::MatrixXd predictionsWithStd(15, 2);
    predictionsWithStd = gp.predict(x_test);
    cout << "debug 5" << endl;

    // 5. Display the predictions.
    std::cout << "Predictions:" << std::endl;
    for (int i = 0; i < 15; ++i) {
        std::cout << "x = " << x_test.row(i) << ", y = " << predictionsWithStd(i,0)
          << "+-" << predictionsWithStd(i,1) << std::endl;
    }

    return 0;
}
