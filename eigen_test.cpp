#include<iostream>
#include<Eigen/Dense>

using namespace std;

int main(){
    Eigen::VectorXd x_train(15), y_train(15);
    x_train<<1,2,3,4,5,6,7,8,9,10,11,12,13,14,15;
    y_train<<1,2,3,4,5,6,7,8,9,10,11,12,13,14,15;
    Eigen::MatrixXd combined_vectors(15, 2);
    combined_vectors << x_train, y_train;
    cout << combined_vectors(10,0) << endl;

    // Append x_train and y_train at the lowest dimension



    return 0;
}
