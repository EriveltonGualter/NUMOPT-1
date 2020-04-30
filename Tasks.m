

% 6.* Implement a program that plots a given R^2?R function as a 3D surface,
% and its second order approximations (Taylor-polynomials) at a given point,
% step-by-step in each step of Newton’s method (without line search and the
% positive definite correction, i.e. based on Newton_noLS.m).

% 9. Implement a non-unimodal line search method, which terminates when a point
% satisfying the Wolfe conditions is found. You may use the code for the Armijo
% conditions (Armijo_LS.m)as reference.

% 12. Implement a modified multivariable Newton method (based onNewton.m),
% which corrects thenon positive definite Hessian by computing its LDL^T 
% decomposition and changing negative and very small diagonal entries to
% some fixed positive ? value.

function [ alpha ] = Armijo_LS(f, df, p, x, alpha, rho, c)
% f: vector->scalar function -- the objective
% df: gradient function
% p: search direction
% x: starting point
% alpha: initial step length
% rho: step lenght multiplier
% c: condition multiplier

    f0 = f(x);
    g0 = df(x);
    x0 = x;
    x = x0 + alpha * p;
    fk = f(x);
    
    % repeat until the Armijo condition is satisfied
    while fk > f0 + c * alpha * (g0'*p)
      alpha = rho * alpha;
      x = x0 + alpha * p;
      fk = f(x);
    end
end