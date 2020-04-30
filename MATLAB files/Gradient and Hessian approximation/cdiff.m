function [ df ] = cdiff(f,x)
% f: R^n->R objective
% x: interior point
% df: estimated gradient using central differentials

    eps = 5e-6;
    n=size(x,1);
    df = zeros(n,1);
    for k=1:n
        dir = zeros(n,1);
        dir(k) = eps;
        fxe = f(x+dir);
        fxrev = f(x-dir);
        df(k) = (fxe-fxrev)/(2*eps);
    end
end