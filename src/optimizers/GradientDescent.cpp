#include "optimizers/GradientDescent.h"
#include <functional>
#include <iostream>
#include <ctime>

GradientDescentOptimizer::GradientDescentOptimizer(
    std::function<double(const Eigen::VectorXd&)> func,
    const Eigen::VectorXd& start_point, // Starting point for optimization
    double lr, // Learning rate
    int iters, // Number of iterations to run
    const Eigen::VectorXd& lower_bounds, // Lower bounds for each dimension (optional)
    const Eigen::VectorXd& upper_bounds, // Upper bounds for each dimension (optional)
    float eps, // Convergence threshold
    int reps, // Number of times to repeat the optimization
    int step, // Number of iterations between learning rate decay
    float kappa, // Decay rate for learning rate
    float h // Step size for numerical gradient
    ) : func(func), point(start_point), learning_rate(lr), iterations(iters),
      epsilon(eps), repeats(reps), step_size(step), decay_rate(kappa),
      lower_bounds(lower_bounds), upper_bounds(upper_bounds), h(h) {}

Eigen::VectorXd GradientDescentOptimizer::minimize() {
    srand((unsigned int)time(NULL));
    Eigen::VectorXd min_point = point;

    for (int i = 0; i < repeats; ++i) {
        if (i > 0)
            point = randomizeStartPoint();

        bool within_bounds = true;
        for (int j = 0; j < iterations; ++j) {
            if (!within_bounds)
                break;

            if (j % step_size == 0 && j > 0)
                learning_rate *= decay_rate;

            Eigen::VectorXd grad = numericalGradient(func, point);

            if (grad.norm() < epsilon)
                break;

            within_bounds = checkBounds(point);

            point -= learning_rate * grad;
            if (func(point) < func(min_point)){
                min_point = point;
                std::cout << "New minimum point: " << min_point.transpose() << " value " << func(min_point) << std::endl;
            }
        }
    }

    return min_point;
}

Eigen::VectorXd GradientDescentOptimizer::numericalGradient(std::function<double(const Eigen::VectorXd&)> func, const Eigen::VectorXd& v) {
    double h = 1e-5;
    Eigen::VectorXd grad(v.size());
    for (int i = 0; i < v.size(); ++i) {
        Eigen::VectorXd v_plus = v;
        v_plus[i] += h;
        grad[i] = (func(v_plus) - func(v)) / h;
    }
    return grad;
}

Eigen::VectorXd GradientDescentOptimizer::randomizeStartPoint() {
    if (lower_bounds.size() > 0 && upper_bounds.size() > 0)
        return lower_bounds + Eigen::VectorXd::Random(lower_bounds.size()).cwiseAbs().cwiseProduct(upper_bounds - lower_bounds);
    else
        return point + 100.*Eigen::VectorXd::Random(point.size());
}

bool GradientDescentOptimizer::checkBounds(const Eigen::VectorXd& pt) {
    if (lower_bounds.size() > 0 && upper_bounds.size() > 0) {
        for (int i = 0; i < pt.size(); ++i){
            if (pt[i] < lower_bounds[i] || pt[i] > upper_bounds[i])
                return false;
        }
    }
    return true;
}
