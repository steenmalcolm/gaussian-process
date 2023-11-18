#include "GradientDescent.h"
#include <iostream>
#include <ctime>

GradientDescentOptimizer::GradientDescentOptimizer(
    double (*func)(const Eigen::VectorXd&),
    const Eigen::VectorXd& start_point,
    double lr,
    int iters,
    const Eigen::VectorXd& lower_bounds,
    const Eigen::VectorXd& upper_bounds,
    float eps,
    int reps,
    int step,
    float kappa)
    : func(func), point(start_point), learning_rate(lr), iterations(iters),
      epsilon(eps), repeats(reps), step_size(step), decay_rate(kappa),
      lower_bounds(lower_bounds), upper_bounds(upper_bounds) {}

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
        }

        if (func(point) < func(min_point)) {
            min_point = point;
            std::cout << "New minimum point: " << min_point.transpose() << " value " << func(min_point) << std::endl;
        }
    }

    return min_point;
}

Eigen::VectorXd GradientDescentOptimizer::numericalGradient(double (*func)(const Eigen::VectorXd&), const Eigen::VectorXd& v) {
    double h = 1e-5;
    Eigen::VectorXd grad(v.size());
    Eigen::VectorXd v_plus = v;
    for (int i = 0; i < v.size(); ++i) {
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
        for (int i = 0; i < pt.size(); ++i)
            if (pt[i] < lower_bounds[i] || pt[i] > upper_bounds[i])
                return false;
    }
    return true;
}
