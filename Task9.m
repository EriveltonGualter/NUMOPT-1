% 9. Implement a non-unimodal line search method, which terminates when a point
% satisfying the Wolfe conditions is found. You may use the code for the Armijo
% conditions (Armijo_LS.m)as reference.

% USAGE: press or hold any key until the condition is satisfied
% if we start with a too big alpha, it gets smaller, it we start with a too
% small one, it will get larger.

f = @(x) 4*sin(x) - x;
df = @(x) 4*cos(x) - 1;
x_start = rand(1) * 2;
alpha = 0.1;
changeRatio = 0.1;
c1 = 0.9;
c2 = 0.95;

hold on
plot(linspace(-4, 6, 500), f(linspace(-4, 6, 500)), 'r');
scatter(x_start, f(x_start), 200, 'b', 'x');

stepsize = Wolfe_LS_WithVisu(f, df, 2*(df(x_start) < 0) - 1, ...
                             x_start, alpha, changeRatio, c1, c2);


function [ alpha ] = Wolfe_LS_WithVisu(f, df, dir, x, alpha, changeRatio, c1, c2)
    % f: vector->scalar function -- the objective
    % df: gradient function
    % dir: search direction
    % x: starting point
    % alpha: initial step length
    % changeRatio: portion of alpha, with which we modify alpha
    % c1: f(x + alpha*dir) shoule be <= f(x) + c1*alpha*dir*df(x)
    % c2: dir*df(x + alpha*dir) should be >= c2*dir*df(x)
    
    slopeHere = dir*df(x)
    fValueHere = f(x);
    
    isTooBig = true;
    isTooSmall = true;
    
    while isTooSmall || isTooBig
        xNext = x + alpha*dir;
        h = scatter(xNext, f(xNext), 200, 'k', 'x');
        
        lineValueThere = fValueHere + alpha*slopeHere*c1;
        fValueThere = f(xNext);
        isTooBig = fValueThere > lineValueThere;
        
        slopeThere = dir*df(xNext)
        isTooSmall = slopeThere < slopeHere*c2;
        
        while waitforbuttonpress == 0
        end
        
        if isTooSmall && isTooBig
            err = 'Cannot satisfy Wolfe'
            break;
        elseif isTooSmall
            alpha = alpha*(1 + changeRatio);
            delete(h)
        elseif isTooBig
            alpha = alpha*(1 - changeRatio);
            delete(h)
        end
    end
    done = 'Wolfe satisfied'
end


function [ alpha ] = Wolfe_LS(f, df, dir, x, alpha, changeRatio, c1, c2)
    % f: vector->scalar function -- the objective
    % df: gradient function
    % dir: search direction
    % x: starting point
    % alpha: initial step length
    % changeRatio: portion of alpha, with which we modify alpha
    % c1: f(x + alpha*dir) shoule be <= f(x) + c1*alpha*dir*df(x)
    % c2: dir*df(x + alpha*dir) should be >= c2*dir*df(x)
    
    slopeHere = dir*df(x);
    fValueHere = f(x);
    
    isTooBig = true;
    isTooSmall = true;
    
    while isTooSmall || isTooBig
        xNext = x + alpha*dir;
        
        lineValueThere = fValueHere + alpha*slopeHere*c1;
        fValueThere = f(xNext);
        isTooBig = fValueThere > lineValueThere;
        
        slopeThere = dir*df(xNext);
        isTooSmall = slopeThere < slopeHere*c2;
        
        if isTooSmall && isTooBig
            break;
        elseif isTooSmall
            alpha = alpha*(1 + changeRatio);
        elseif isTooBig
            alpha = alpha*(1 - changeRatio);
        end
    end
end
