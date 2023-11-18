#ifndef GRADIENT_DESCENT_OPTIMIZER_H
#define GRADIENT_DESCENT_OPTIMIZER_H

#include <Eigen/Dense>

class GradientDescentOptimizer {
public:
    GradientDescentOptimizer(
        double (*func)(const Eigen::VectorXd&),
        const Eigen::VectorXd& start_point,
        double lr,
        int iters,
        const Eigen::VectorXd& lower_bounds = Eigen::VectorXd(),
        const Eigen::VectorXd& upper_bounds = Eigen::VectorXd(),
        float eps = 1e-6,
        int reps = 1000,
        int step = 100,
        float kappa = 0.99);

    Eigen::VectorXd minimize();

private:
    double (*func)(const Eigen::VectorXd&);
    Eigen::VectorXd point;
    double learning_rate;
    int iterations;
    float epsilon;
    int repeats;
    int step_size;
    float decay_rate;
    Eigen::VectorXd lower_bounds, upper_bounds;

    Eigen::VectorXd numericalGradient(double (*func)(const Eigen::VectorXd&), const Eigen::VectorXd& v);
    Eigen::VectorXd randomizeStartPoint();
    bool checkBounds(const Eigen::VectorXd& pt);
};

#endif // GRADIENT_DESCENT_OPTIMIZER_H
