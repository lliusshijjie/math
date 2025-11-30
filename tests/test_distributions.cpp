#include <gtest/gtest.h>
#include <math/probability/distributions.hpp>
#include <math/core/constants.hpp>

using namespace math::probability;
using namespace math::core;

TEST(DistributionTest, Uniform) {
    // PDF: 1/(b-a) inside [a, b]
    EXPECT_NEAR(Uniform<double>::pdf(0.5, 0.0, 1.0), 1.0, 1e-9);
    EXPECT_NEAR(Uniform<double>::pdf(1.5, 0.0, 1.0), 0.0, 1e-9);
    
    // CDF: (x-a)/(b-a)
    EXPECT_NEAR(Uniform<double>::cdf(0.5, 0.0, 1.0), 0.5, 1e-9);
    EXPECT_NEAR(Uniform<double>::cdf(1.2, 0.0, 1.0), 1.0, 1e-9);
    
    // Quantile
    EXPECT_NEAR(Uniform<double>::quantile(0.5, 0.0, 1.0), 0.5, 1e-9);
    EXPECT_NEAR(Uniform<double>::quantile(0.25, 0.0, 10.0), 2.5, 1e-9);
}

TEST(DistributionTest, Normal) {
    // PDF at mu should be 1/(sigma*sqrt(2pi))
    double pdf_at_mu = 1.0 / std::sqrt(2.0 * PI);
    EXPECT_NEAR(Normal<double>::pdf(0.0, 0.0, 1.0), pdf_at_mu, 1e-9);
    
    // CDF at mu should be 0.5
    EXPECT_NEAR(Normal<double>::cdf(0.0, 0.0, 1.0), 0.5, 1e-9);
    
    // CDF at 1 sigma (approx 0.8413)
    EXPECT_NEAR(Normal<double>::cdf(1.0, 0.0, 1.0), 0.8413447, 1e-6);
    
    // Quantile
    EXPECT_NEAR(Normal<double>::quantile(0.5, 0.0, 1.0), 0.0, 1e-6);
    // Quantile 0.975 is approx 1.96
    EXPECT_NEAR(Normal<double>::quantile(0.975, 0.0, 1.0), 1.95996, 1e-4);
}

TEST(DistributionTest, Exponential) {
    // PDF: lambda * exp(-lambda * x)
    EXPECT_NEAR(Exponential<double>::pdf(1.0, 1.0), std::exp(-1.0), 1e-9);
    
    // CDF: 1 - exp(-lambda * x)
    EXPECT_NEAR(Exponential<double>::cdf(1.0, 1.0), 1.0 - std::exp(-1.0), 1e-9);
    
    // Quantile
    // p = 1 - exp(-lambda * x) => exp(-lambda * x) = 1 - p => -lambda * x = ln(1-p) => x = -ln(1-p)/lambda
    EXPECT_NEAR(Exponential<double>::quantile(0.5, 1.0), -std::log(0.5), 1e-9);
}

TEST(DistributionTest, Binomial) {
    // PMF(k, n, p)
    // k=0: (1-p)^n
    EXPECT_NEAR(Binomial<double>::pmf(0, 5, 0.5), std::pow(0.5, 5), 1e-9);
    // k=n: p^n
    EXPECT_NEAR(Binomial<double>::pmf(5, 5, 0.5), std::pow(0.5, 5), 1e-9);
    
    // n=2, p=0.5. PMF: 0->0.25, 1->0.5, 2->0.25
    EXPECT_NEAR(Binomial<double>::pmf(1, 2, 0.5), 0.5, 1e-9);
    
    // CDF
    EXPECT_NEAR(Binomial<double>::cdf(0, 2, 0.5), 0.25, 1e-9);
    EXPECT_NEAR(Binomial<double>::cdf(1, 2, 0.5), 0.75, 1e-9);
    EXPECT_NEAR(Binomial<double>::cdf(2, 2, 0.5), 1.0, 1e-9);
}

TEST(DistributionTest, Poisson) {
    // PMF(k, lambda) = lambda^k * e^-lambda / k!
    // k=0: e^-lambda
    EXPECT_NEAR(Poisson<double>::pmf(0, 1.0), std::exp(-1.0), 1e-9);
    
    // k=1: lambda * e^-lambda
    EXPECT_NEAR(Poisson<double>::pmf(1, 2.0), 2.0 * std::exp(-2.0), 1e-9);
    
    // CDF sum
    double cdf0 = std::exp(-1.0);
    double cdf1 = cdf0 + std::exp(-1.0); // lambda=1, k=0,1 terms are same
    EXPECT_NEAR(Poisson<double>::cdf(0, 1.0), cdf0, 1e-9);
    EXPECT_NEAR(Poisson<double>::cdf(1, 1.0), cdf1, 1e-9);
}
