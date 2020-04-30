function [ x ] = test_method(G,f,df,ddf,testpts,startpt,a,b,c,d,res)
% G: optimization method to test; required signature: (f,df,ddf,x0,iter)
% f: R^2->R objective
% df: gradient
% ddf: Hessian
% testpts: number of iterations for G
% startpt: starting point
% a,b,c,d: rectangular region to plot = [a,b] x [c,d]
% res: plot resolution = res x res
% x: output after each iteration

    X = linspace(a,b,res);
    Y = linspace(c,d,res);
    Z = zeros(res,res);
    for i=1:res
        for j=1:res
            Z(i,j)=f([X(i);Y(j)]);
        end
    end

    x = zeros(testpts,2);
    x(1,:) = startpt;
    for k=2:testpts
        x(k,:) = G(f,df,ddf,[x(1,1);x(1,2)],k-1);
    end
    contourf(X,Y,Z',200); hold on
    scatter(x(:,1),x(:,2),'r','filled');
    hold off;
end
