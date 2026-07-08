/**
 * @file black_scholes.cpp
 * @brief Black-Scholes pricing implementation.
 *
 * Closed-form European option pricing and Greeks via the Black-Scholes model.
 * Handles the expired case (T <= 0) by returning intrinsic value only.
 */

#include "quantedge/black_scholes.hpp"
#include <cmath>
#include <algorithm>

namespace quantedge {

    // Standard normal probability density function
    static double norm_pdf(double x) {
        static const double INV_SQRT_2PI = 0.3989422804014327;
        return INV_SQRT_2PI * std::exp(-0.5 * x * x);
    }

    // Standard normal cumulative distribution function via complementary error function.
    // erfc(-x / sqrt(2)) / 2  ==>  multiply x by 1/sqrt(2) = M_SQRT1_2
    static double norm_cdf(double x) {
        return 0.5 * std::erfc(-x * M_SQRT1_2);
    }

BSResult black_scholes(double S,double K,double T,double r,double sigma,OptionType type){
    BSResult result{};

    if(T<=0){
        if(type==OptionType::Call){
            result.price=std::max(0.0,S-K);
            result.delta=S>K?1.0:0.0;
        }else{
            result.price=std::max(0.0,K-S);
            result.delta=S<K?-1.0:0.0;
        }

        result.gamma=0.0;
        result.vega=0.0;
        result.theta=0.0;
        result.rho=0.0;
        return result;
    }


    double sqrt_T=std::sqrt(T);
    double d1=(std::log(S/K)+(r+0.5*sigma*sigma)*T)/(sigma*sqrt_T);
    double d2=d1-sigma*sqrt_T;
    
    double Nd1    = norm_cdf(d1);
    double Nd2    = norm_cdf(d2);
    double Nnd1   = norm_cdf(-d1);
    double Nnd2   = norm_cdf(-d2);
    double pdf_d1 = norm_pdf(d1);
    double exp_rT = std::exp(-r * T);

    if (type == OptionType::Call) {
        result.price = S * Nd1 - K * exp_rT * Nd2;
        result.delta = Nd1;
        result.theta = -(S * pdf_d1 * sigma) / (2.0 * sqrt_T)
                       - r * K * exp_rT * Nd2;
        result.rho   = K * T * exp_rT * Nd2;
    } else {
        result.price = K * exp_rT * Nnd2 - S * Nnd1;
        result.delta = Nd1 - 1.0;
        result.theta = -(S * pdf_d1 * sigma) / (2.0 * sqrt_T)
                       + r * K * exp_rT * Nnd2;
        result.rho   = -K * T * exp_rT * Nnd2;
    }
    
    result.gamma = pdf_d1 / (S * sigma * sqrt_T);
    result.vega  = S * pdf_d1 * sqrt_T;
    return result;

  }

}