function [ df ] = fdiff(f,x)
% f: R^n->R objective
% x: interior point
% df: estimated gradient using forward differentials

    eps = 15e-9;
    n=size(x,1);
    df = zeros(n,1);
    fx = f(x);
    for k=1:n
        dir = zeros(n,1);
        dir(k) = eps;
        fxe = f(x+dir);
        df(k) = (fxe-fx)/eps;
    end
end