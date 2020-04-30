function [ x ] = Newton_noLS(~,df,ddf,x0,iter)
% f: vector->scalar objective function; not used!
% df: gradient function
% ddf: Hessian
% x0: starting point
% iter: number of iterations

    x=x0;
    for k=1:iter
        p = -inv(ddf(x))*df(x);
        %?? LineSearch
        x = x+p;
    end
end
