function x = Newton_unimodal(~,df,ddf,x0,~,~,iter)
% f: real->real objective function, not needed
% df: derivative
% ddf: second derivative
% x0: starting point
% iter: number of iterations

    x=x0;
    for k=1:iter
        dfx=df(x);
        ddfx=ddf(x);
        x=x-dfx/ddfx;
    end
end
