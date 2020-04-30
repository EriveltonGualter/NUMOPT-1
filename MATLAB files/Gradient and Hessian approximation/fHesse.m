function [ ddf ] = fHesse(df,x)
% f: R^n->R objective
% x: interior point
% ddf: estimated Hessian using forward differentials

    eps = 15e-9;
    n=size(x,1);
    H = zeros(n);
    dfx = df(x);
    for k=1:n
        dir = zeros(n,1);
        dir(k) = eps;
        dfxe = df(x+dir);
        H(:,k) = (dfxe-dfx)/eps;
    end
    ddf = (H+H')/2;
end