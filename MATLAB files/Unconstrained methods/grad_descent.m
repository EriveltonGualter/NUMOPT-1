function [ x ] = grad_descent(f,df,~,x0,iter)
% f: vector->scalar objective function
% df: gradient function
% x0: starting point
% iter: number of iterations

    rho=0.5; %For the Armijo LS usually 0.5 or 0.9
    c=0.2; %For the Armijo LS usually 0.01 or 0.2
    alpha=1; %For the Armijo LS
	
    x=x0;
    for k=1:iter
        p=-df(x); %Direction of (steepest) descent
        p=p/norm(p); %Normalized
        gamma=Armijo_LS(f,df,p,x,alpha,rho,c); %Compute step length
        x=x+gamma*p;
    end
end
