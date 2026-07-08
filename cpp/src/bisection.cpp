/**
 * @file bisection.cpp
 * @brief Bisection method for implied volatility.
 *
 * TODO: Implement IV solver using bisection.
 */

#include "quantedge/iv_solver.hpp"
#include <cmath>
#include <algorithm>

namespace quantedge {

double iv_bisection(double observed_price,double S,double K,double T,double r,OptionType type,const IVSolverConfig& config){
   
    if(T<=0)return 0.0;

    double low=config.vol_lower;
    double high=config.vol_upper;

    for(int i=0;i<config.max_iter;++i){
        double mid=(low+high)/2.0;

        if(mid < 1e-6)mid=1e-6;

        BSResult bs=black_scholes(S,K,T,r,mid,type);

        if(std::abs(bs.price-observed_price)<config.tolerance){
            return mid;
        }

        if(bs.price<observed_price){
            low=mid;
        }else{
            high=mid;
        }
       
        if((high-low)<config.tolerance){
            return mid;
        }
    }
    return (low+high)/2.0;
}
} // namespace quantedge
