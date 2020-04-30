% 2. Suppose that there are some given number m?N of immobile electric charges
% at some points of the plane. We want to place some given n?N number of further
% charged particles (not including the fixed charges) along a circle centered
% at the origin, in such a way that the sumpotential of the system is minimized.
% Supposing that all charges are equal, formalize this optimization problem.
% Which known algorithm would you recommend to solve the problem with?
% Demonstrate an approximate solution for a parameter setting of your choice.

% =========================================================================
% parameter setting
clc, clear all
global fixedPoints
global radius
%fixedPoints = [[0, 10]; [10, 0]; [0, -10]; [-10, 0]];[[0, 10]; [10, 0]; [0, -10]; [-10, 0]];
fixedPoints = [0, 0];
radius = 5;
numPointsToPlace = 2;

% =========================================================================
% create initial random values and plot them
%fi0 = 2 * pi * rand(numPointsToPlace, 1);
fi0 = [0; pi]
plotState(fi0)

% iterate, and continously plot the results
optimizeWithNewtonAndPlot(fi0, 50)



function loss = calcPotentailSum(fi)
    global fixedPoints
    global radius
    loss = 0;
    % loop over each point, and calculate
    for varId = 1 : size(fi, 1)
        % force with fixed points
        % 1 / dist([r * cos(fi), r * sin(fi)], [Px, Py])
        % =
        % 1 / sqrt(r*r*cos^2(fi) - 2*Px*r*cos(fi) + Px^2 +
        %          r*r*sin^2(fi) - 2*Py*r*sin(fi) + Py^2)
        % =
        % 1 / sqrt(r*r + Px*Px + Py*Py -2*r*(Px*cos(fi) + Py*sin(fi))
        for fixId = 1 : size(fixedPoints, 1)
            cosfi = cos(fi(varId));
            sinfi = sin(fi(varId));
            p2x = fixedPoints(fixId, 1);
            p2y = fixedPoints(fixId, 2);
            loss = loss + 1 ...
                / sqrt(radius*radius + p2x*p2x + p2y*p2y ...
                       - 2*radius*(p2x*cosfi + p2y*sinfi));        
        end
        % force with other variable points
        % 1 / dist([r*cos(fi1), r*sin(fi1)], [r*cos(fi2), r*sin(fi2)])
        % = 
        % 1 / sqrt(r*r + r*r - 2*r*r*cos(fi2)*cos(fi1) - 2*r*r*sin(fi2)*sin(fi1) 
        % =
        % 1 / sqrt(2*r*r*(1 - cos(fi1)*cos(fi2) - sin(fi1)*sin(fi2))
        for otherVarId = fixId + 1 : size(fi, 1)
            cos1 = cos(fi(varId));
            sin1 = sin(fi(varId));
            cos2 = cos(fi(otherVarId));
            sin2 = sin(fi(otherVarId));
            loss = loss + 1 / sqrt(2*radius*radius*(1 - cos1*cos2 - sin1*sin2));   
        end
    end
end

function lossGrad = calcPotentialGrad(fi)
    global fixedPoints
    global radius
    % derivate of 1/sqrt(f(x)) = -1 * f'(x) / (2*(f(x)^(3/2)))
    lossGrad = zeros(size(fi,1 ), 1);
    for varId = 1 : size(fi, 1)
        % we only take into account the forces in connection with this point
        % with fixed points, we diff
        % 1 / sqrt(r*r + Px*Px + Py*Py -2*r*(Px*cos(fi) + Py*sin(fi))
        % which will be
        % r*(Px*-1*sin(fi) + Py*cos(fi)) / 
        % (r*r + Px*Px + Py*Py -2*r*(Px*cos(fi) + Py*sin(fi))^(3/2)
        for fixId = 1 : size(fixedPoints, 1)
            cosfi = cos(fi(varId));
            sinfi = sin(fi(varId));
            p2x = fixedPoints(fixId, 1);
            p2y = fixedPoints(fixId, 2);
            nomi = radius*(p2y*cosfi - p2x*sinfi);
            denomi = radius*radius + p2x*p2x + p2y*p2y - 2*radius*(p2x*cosfi + p2y*sinfi);
            denomi = denomi^(3/2);
            lossGrad(varId) = lossGrad(varId) + nomi / denomi;
        end
        % with other points, we diff
        % 1 / sqrt(2*r*r*(1 - cos(fi1)*cos(fi2) - sin(fi1)*sin(fi2))
        % which will be
        % -r*r*(sin(fi1)*cos(fi2) - cos(fi1)*sin(fi2)) / 
        % (2*sqrt(2)*(r*r*(1 - cos(fi1)*cos(fi2) - sin(fi1)*sin(fi2)))^(3/2)
        % but in practice we will use the same equation as before
        % only p2 constant is different
        for otherVarId = 1 : size(fi, 1)
            if otherVarId ~= varId
                cos1 = cos(fi(varId));
                sin1 = sin(fi(varId));
                cos2 = cos(fi(otherVarId));
                sin2 = sin(fi(otherVarId));
                nomi = -1*radius*radius*(sin1*cos2 - cos1*sin2);
                denomi = 2*sqrt(2)*(radius*radius*(1 - cos1*cos2 - sin1*sin2))^(3/2);
                lossGrad(varId) = lossGrad(varId) + nomi / denomi;
            end
        end
    end
end

function lossHessian = calcPotentialHessian(fi)
    global radius
    % basically each element of the hessian is only dependent on the
    % force between the given two points which is
    % 1 / sqrt(cos^2(fi1) - 2*cos(fi2)*cos(fi1) + cos^2(fi2) +
    %          sin^2(fi1) - 2*sin(fi2)*sin(fi1) + sin^2(fi2))
    % diffed along fi1, it looks like : f(fi2)*g(fi2)^(-3/2)
    % f(fi2) = -r*r(sin(fi1)*cos(fi2) - cos(fi1)*sin(fi2)) 
    % g(fi2) = 2*r*r*(1 - cos(fi1)*cos(fi2) - sin(fi1)*sin(fi2)
    % derivative of f(x)*g(x)^(-3/2) =
    % (2*g(x)*f'(x) - 3*f(x)*g'(x)) / (2*g(x)^(5/2))
    lossHessian = zeros(size(fi, 1));
    for id1 = 1 : size(fi, 1)
        for id2 = id1 + 1 : size(fi)
            cos1 = cos(fi(id1));
            sin1 = sin(fi(id1));
            cos2 = cos(fi(id2));
            sin2 = sin(fi(id2));
            f  = -1*radius*radius*(sin1*cos2 - cos1*sin2);
            df = radius*radius*(sin1*sin2 + cos1*cos2);
            g  = 2*radius*radius*(1 - cos1*cos2 - sin1*sin2);
            dg = 2*radius*radius*(cos1*sin2 - sin1*cos2);
            result = (2*g*df - 3*f*dg) / (2*g^(5/2));
            lossHessian(id1, id2) = result;
            lossHessian(id2, id1) = result;
        end
    end
end

function plotState(fi)
    global fixedPoints
    global radius
    hold on
    circleX = radius * cos(0: pi / 50: 2 * pi);
    circleY = radius * sin(0: pi / 50: 2 * pi);
    plot(circleX, circleY, 'r')
    scatter(fixedPoints(:, 1), fixedPoints(:, 2), 500, 'b', '.')
    scatter(radius * cos(fi), radius * sin(fi), 400, 'r', '.')
    pause
    clf
end

function [ x ] = optimizeWithNewtonAndPlot(x0, iter)
% f: vector->scalar objective function
% df: gradient function
% ddf: Hessian
% x0: starting point
% iter: number of iterations

    rho = 0.5; %For the Armijo LS; usually 0.5 or 0.9
    c = 0.2; %For the Armijo LS; usually 0.01 or 0.2
    
    delta = 0.00000001; %Correction constant to ensure p is a descent direction

    x = x0;
    for k = 1 : iter
        H = calcPotentialHessian(x);
        [Q, D] = eig(H);
        for i = 1 : size(D(:, 1))
            if D(i, i) < delta
                D(i, i) = 1 / delta;
            else
                D(i, i) = 1 / D(i, i);
            end
        end
        dir = -Q * D * Q' * calcPotentialGrad(x);
        alpha = norm(dir);
        dir = dir / norm(dir);
        if sum(isnan(dir)) > 0
            break;
        end
        gamma = Armijo_LS(dir, x, alpha, rho, c); %Compute step length
        sprintf('step length is %f', gamma)
        x = x + gamma * dir;
        
        plotState(x)
    end
end

function [ alpha ] = Armijo_LS(dir, x, alpha, rho, c)
% f: vector->scalar function -- the objective
% df: gradient function
% p: search direction
% x: starting point
% alpha: initial step length
% rho: step lenght multiplier
% c: condition multiplier

    f0 = calcPotentailSum(x);
    g0 = calcPotentialGrad(x);
    x0 = x;
    x = x0 + alpha * dir;
    fk = calcPotentailSum(x);
    
    % repeat until the Armijo condition is satisfied
    while fk > f0 + c * alpha * (g0' * dir)
      alpha = rho * alpha;
      x = x0 + alpha * dir;
      fk = calcPotentailSum(x);
    end
end











