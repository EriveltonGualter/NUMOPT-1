% 6.* Implement a program that plots a given R^2?R function as a 3D surface,
% and its second order approximations (Taylor-polynomials) at a given point,
% step-by-step in each step of Newton’s method (without line search and the
% positive definite correction, i.e. based on Newton_noLS.m).

% USAGE: PRESS ANY KEY TO STEP. WHEN STEPPING, TURN OFF THE DRAGGING MODE
% ELSE IT WILL FLASH

%% PARAMETERS =============================================================
% Sample Function
f=@(x) 100*(x(2)-x(1)^2)^2+(1-x(1))^2;
df=@(x) [400*x(1)^3-400*x(1)*x(2)+2*x(1)-2;200*(x(2)-x(1)^2)];
ddf=@(x) [1200*x(1)^2-400*x(2)+2 -400*x(1);-400*x(1) 200];

% taylor general
t2 = @(x0, x) f(x0) + df(x0)'*(x-x0) + (x-x0)'*ddf(x0)*(x-x0);

% Region and Resolution
startX = -1;
endX = 1.2;
startY = -0.2;
endY = 1.2;
res = 100;

% number of iterations to go upto
maxIter = 500;

%% SETTING UP STARTING STATE ==============================================
x = rand(2, 1);
x(1) = x(1) * (endX - startX) + startX;
x(2) = x(2) * (endY - startY) + startY;

X = linspace(startX, endX, res);
Y = linspace(startY, endY, res);
Z = zeros(res, res);
T = zeros(res, res);

for i=1:res
    for j=1:res
        Z(j, i) = f([X(i); Y(j)]);
        T(j, i) = t2(x, [X(i); Y(j)]);
    end
end

% starting plot
surf(X, Y, Z, 'EdgeColor', 'none');
zlim([min(Z(:)), max(Z(:))])
hold on
hPoint = scatter3(x(1), x(2), f(x), 'r','filled');
hTaylor = surf(X, Y, T);

while waitforbuttonpress == 0
end

%% main loop ==============================================================
for iterIndex = 1 : maxIter
    x = Newton_noLS(f, df, ddf, x, 1);

    for i=1:res
        for j=1:res
            T(j, i) = t2(x, [X(i); Y(j)]);
        end
    end
    
    delete(hPoint)
    delete(hTaylor)
    hPoint = scatter3(x(1), x(2), f(x), 'r','filled');
    hTaylor = surf(X, Y, T);
    
    while waitforbuttonpress == 0
    end
end

%% Functions
function [ x ] = Newton_noLS(~,df,ddf,x0,iter)
    % f: vector->scalar objective function; not used!
    % df: gradient function
    % ddf: Hessian
    % x0: starting point
    % iter: number of iterations
    x = x0;
    for k = 1: iter
        p = -inv(ddf(x)) * df(x);
        x = x + 0.25 * p;
    end
end
