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
