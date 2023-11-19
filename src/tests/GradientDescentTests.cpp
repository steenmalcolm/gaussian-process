// Class that implements the unit tests for the GradientDescent class.  
#include <iostream>
#include "GradientDescent.h"
#include <Eigen/Dense>

class GradientDescentTests {
private:
    Eigen::VectorXd some_data;
    Eigen::VectorXd start_point;
    Eigen::VectorXd lower_bounds, upper_bounds;
    double learning_rate;
    int iterations;

    
    // return a rugged function with many local minima
    double func(const Eigen::VectorXd& v) {
        double result = 0.0;
        for (int i = 0; i < some_data.size(); ++i) 
            result += some_data[i] * std::sin(v[0] * i + v[1]);
        return result;
    }

    void brute_force_minimizer(){
        double min = 1e10;
        double min_x = 0;
        double min_y = 0;
        for (double x = lower_bounds[0]; x < upper_bounds[0]; x += 0.01) {
            for (double y = lower_bounds[1]; y < upper_bounds[1]; y += 0.01) {
                Eigen::VectorXd v(2);
                v << x, y;
                double val = func(v);
                if (val < min) {
                    min = val;
                    min_x = x;
                    min_y = y;
                }
            }
        }
        std::cout << "Brute force minimum point: " << min_x << " " << min_y << " value " << min << std::endl;
    }

public:
    GradientDescentTests() 
        : some_data(Eigen::VectorXd::Random(10)),
            start_point(Eigen::VectorXd::Constant(2, 1)),
            lower_bounds(Eigen::VectorXd::Constant(2, -10)),
            upper_bounds(Eigen::VectorXd::Constant(2, 10)),
            learning_rate(0.001),
            iterations(1000)
        {

        }

    void testMinimizer() {
        auto bound_func = std::bind(&GradientDescentTests::func, this, std::placeholders::_1);
        GradientDescentOptimizer optimizer(bound_func, start_point, learning_rate, iterations, lower_bounds, upper_bounds);
        Eigen::VectorXd min_point = optimizer.minimize();
        std::cout << "Minimum point: " << min_point.transpose() << " value " << func(min_point) << std::endl;
        brute_force_minimizer();
    }
};


int main() {
    GradientDescentTests tests;

    tests.testMinimizer();
    return 0;
}