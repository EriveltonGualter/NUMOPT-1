function [ x ] = QN_rank1(f,df,~,x0,iter)
% f: vector->scalar objective
% df: derivative
% x0: starting point
% iter: number of iterations

    eps=1e-10;
    n=size(x0,1);
    S=eye(n,n);
    
    x=x0;
	g=df(x);
	p=-g;
    for k=1:iter
        if norm(g)>eps
            fdir=@(t) f(x+t*p);
            tmax=eps;
            while fdir(tmax)<fdir(tmax/2)
                tmax=2*tmax;
            end
            gamma=zero_unimodal(fdir,0,0,0,0,tmax,2*log(eps/tmax)/log(2/3)); %Wolfe LS!!
            p=gamma*p;
            x=x+p;
            q=-g;
            g=df(x);
            q=g+q;
            diff = p-S*q;
            if norm(diff)>eps
                if abs(diff'*q)<eps
                    break;
                end
                beta=diff'*q;
                S=S+(1/beta).*(diff*diff');
            end
            p=-S*g;
        end
    end
end