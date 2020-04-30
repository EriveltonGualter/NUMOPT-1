function [ x ] = Newton(f,df,ddf,x0,iter)
% f: vector->scalar objective function; not used!
% df: gradient function
% ddf: Hessian
% x0: starting point
% iter: number of iterations

    rho=0.5; %For the Armijo LS; usually 0.5 or 0.9
    c=0.2; %For the Armijo LS; usually 0.01 or 0.2
    
    delta = 0.00000001; %Correction constant to ensure p is a descent direction

    x=x0;
    for k=1:iter
        H = ddf(x);
        [Q,D] = eig(H);
        for i=1:size(D(:,1))
            if D(i,i)<delta
                D(i,i)=1/delta;
            else
                D(i,i) = 1/D(i,i);
            end
        end
        p = -Q*D*Q'*df(x);
        alpha = norm(p);
        p = p/norm(p);
        if sum(isnan(p)) > 0
            break;
        end
        gamma = Armijo_LS(f,df,p,x,alpha,rho,c); %Compute step length
        x = x+gamma*p;
    end
end
