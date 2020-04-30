function [ x ] = FR_conj_grad(f,df,~,x0,iter)
% f: vector->scalar objective
% df: derivative
% x0: starting point
% iter: number of iterations

    eps=1e-10;
    
    for k=1:iter
        x=x0;
        g=df(x);
        p=-g;
        for i=1:size(x)
            if norm(g)>eps
                fdir=@(t) f(x+t*p);
                tmax=eps;
                while fdir(tmax)<fdir(tmax/2)
                    tmax=2*tmax;
                end
                gamma=zero_unimodal(fdir,0,0,0,0,tmax,2*log(eps/tmax)/log(2/3));
                x=x+gamma*p;
                beta=1/(g'*g);
                g=df(x);
                beta=beta*(g'*g);
                p=-g+beta*p;
                if p'*g>0
                    p = -p;
                end
            end
        end
        x0=x;
    end
end