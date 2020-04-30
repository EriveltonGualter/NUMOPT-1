function x = Armijo_LS_basic(f, df, ~, x0, ~, ~, iter)
% f: real->real function -- the objective
% df: derivative
% x0: starting point

    alpha = 1; %initial step length
    rho = 0.5; %step length multiplier
    c = 0.1; %Armijo condition modifier
    
    for k=1:iter
        alpha = 1;
        f0 = f(x0);
        df0 = df(x0);
        if df0 > 0
            alpha = -alpha;
        end
        x = x0+alpha;
        fx = f(x);
    
        while fx > f0 + c*alpha*df0
            alpha = rho*alpha;
            x = x0+alpha;
            fx = f(x);
        end
        x0 = x;
    end
end