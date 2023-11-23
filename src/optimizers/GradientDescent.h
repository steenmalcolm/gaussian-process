#ifndef GRADIENT_DESCENT_OPTIMIZER_H
#define GRADIENT_DESCENT_OPTIMIZER_H

#include <Eigen/Dense>
#include <functional>

class GradientDescentOptimizer {
public:
    GradientDescentOptimizer(
        std::function<double(const Eigen::VectorXd&)> func,
        const Eigen::VectorXd& start_point,
        double lr,
        int iters,
        const Eigen::VectorXd& lower_bounds = Eigen::VectorXd(),
        const Eigen::VectorXd& upper_bounds = Eigen::VectorXd(),
        float eps = 1e-6,
        int reps = 1000,
        int step = 100,
        float kappa = 0.99,
        float h = 1e-5);
    // std::function<double (Eigen::Matrix<double, -1, 1, 0, -1, 1> const&)>
    // Eigen::Matrix<double, -1, 1, 0, -1, 1> const&
    // double
    // int
    // Eigen::Matrix<double, -1, 1, 0, -1, 1> const&
    // Eigen::Matrix<double, -1, 1, 0, -1, 1> const&
    // float
    // int
    // int
    // float
    // float

    Eigen::VectorXd minimize();

private:
    std::function<double(const Eigen::VectorXd&)> func;
    Eigen::VectorXd point;
    double learning_rate;
    int iterations;
    float epsilon;
    int repeats;
    int step_size;
    float decay_rate;
    Eigen::VectorXd lower_bounds, upper_bounds;
    float h;

    Eigen::VectorXd numericalGradient(std::function<double(const Eigen::VectorXd&)> func, const Eigen::VectorXd& v);
    Eigen::VectorXd randomizeStartPoint();
    bool checkBounds(const Eigen::VectorXd& pt);
};

#endif // GRADIENT_DESCENT_OPTIMIZER_H
