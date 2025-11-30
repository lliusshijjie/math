#include <gtest/gtest.h>
#include <math/probability/statistics.hpp>
#include <vector>
#include <array>

using namespace math::probability;

TEST(StatisticsTest, Descriptive) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    // Mean
    EXPECT_NEAR(mean(data.begin(), data.end()), 3.0, 1e-9);
    
    // Variance (sample) -> sum((x-3)^2)/(5-1) = (4+1+0+1+4)/4 = 10/4 = 2.5
    EXPECT_NEAR(variance(data.begin(), data.end()), 2.5, 1e-9);
    
    // Variance (population) -> 10/5 = 2.0
    EXPECT_NEAR(variance_population(data.begin(), data.end()), 2.0, 1e-9);
    
    // Stddev
    EXPECT_NEAR(stddev(data.begin(), data.end()), std::sqrt(2.5), 1e-9);
    
    // Median
    EXPECT_NEAR(median(data.begin(), data.end()), 3.0, 1e-9);
    
    // Even number of elements for median
    std::vector<double> data2 = {1.0, 2.0, 3.0, 4.0};
    EXPECT_NEAR(median(data2.begin(), data2.end()), 2.5, 1e-9);
    
    // Quantile
    // 0.5 should be median
    EXPECT_NEAR(quantile(data.begin(), data.end(), 0.5), 3.0, 1e-9);
}

TEST(StatisticsTest, Correlation) {
    std::vector<double> x = {1.0, 2.0, 3.0};
    std::vector<double> y = {2.0, 4.0, 6.0};
    
    // Covariance
    // mean_x = 2, mean_y = 4
    // (1-2)(2-4) + (2-2)(4-4) + (3-2)(6-4) = (-1)(-2) + 0 + (1)(2) = 2 + 2 = 4
    // div by n-1 = 2 -> 2.0
    EXPECT_NEAR(covariance(x.begin(), x.end(), y.begin()), 2.0, 1e-9);
    
    // Correlation
    // std_x = 1.0, std_y = 2.0
    // cov / (std_x * std_y) = 2.0 / (1.0 * 2.0) = 1.0
    EXPECT_NEAR(correlation(x.begin(), x.end(), y.begin()), 1.0, 1e-9);
    
    // Uncorrelated
    std::vector<double> y2 = {1.0, 0.0, 1.0}; // symmetric around mean
    // cov should be 0
    EXPECT_NEAR(covariance(x.begin(), x.end(), y2.begin()), 0.0, 1e-9);
}

TEST(StatisticsTest, Regression) {
    // y = 2x + 1
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {3.0, 5.0, 7.0, 9.0, 11.0};
    
    auto result = linear_regression<double>(x.begin(), x.end(), y.begin());
    
    EXPECT_NEAR(result.slope, 2.0, 1e-9);
    EXPECT_NEAR(result.intercept, 1.0, 1e-9);
    EXPECT_NEAR(result.r_squared, 1.0, 1e-9);
    EXPECT_NEAR(result.std_error, 0.0, 1e-9);
    
    // Noisy data
    // x: 1, 2, 3
    // y: 1, 2, 2
    // mean_x=2, mean_y=5/3=1.666
    // dx: -1, 0, 1
    // dy: -0.666, 0.333, 0.333
    // numer: (-1)*(-0.666) + 0 + 1*0.333 = 0.666 + 0.333 = 1
    // denom: 1 + 0 + 1 = 2
    // slope = 0.5
    // intercept = 1.666 - 0.5*2 = 0.666
    std::vector<double> x2 = {1.0, 2.0, 3.0};
    std::vector<double> y2 = {1.0, 2.0, 2.0};
    
    auto res2 = linear_regression<double>(x2.begin(), x2.end(), y2.begin());
    EXPECT_NEAR(res2.slope, 0.5, 1e-9);
    EXPECT_NEAR(res2.intercept, 5.0/3.0 - 1.0, 1e-9);
}
