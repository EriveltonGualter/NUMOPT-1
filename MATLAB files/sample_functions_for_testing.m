%sample1 - unimodal, LineSearch
f=@(x) x.^3-3*x.^2+4;
df=@(x) 3*x.^2-6*x;
ddf=@(x) 6*x-6;

%sample2 - unimodal, LineSearch
f=@(x) x.^4;
df=@(x) 4*x.^3;
ddf=@(x) 12*x.^2;

%sample3 - unimodal, LineSearch
f=@(x) sqrt(1+x.^2);
df=@(x) x/sqrt(1+x.^2);
ddf=@(x) (1/sqrt(1+x.^2)).^3;

%sample4 - gradient descent, multivariable Newton, conjugate gradient;
%recommended region: [-2,2]x[-2,2]
f=@(x) x(1)^4+x(2)^4-4*x(1)^2;
df=@(x) [4*x(1)^3-8*x(1);4*x(2)^3];
ddf=@(x) [12*x(1)^2-8 0;0 12*x(2)^2];

%sample5 - Rosenbrock function for gradient descent, multivariable Newton, conjugate gradient
%recommended region: [-1.0,1.2]x[-0.2,1.2]
f=@(x) 100*(x(2)-x(1)^2)^2+(1-x(1))^2;
df=@(x) [400*x(1)^3-400*x(1)*x(2)+2*x(1)-2;200*(x(2)-x(1)^2)];
ddf=@(x) [1200*x(1)^2-400*x(2)+2 -400*x(1);-400*x(1) 200];

%sample6 - the "ideal" convex paraboloid(s)
%recommended region: [-2,2]x[-2,2]
f=@(x) 0.01*x(1)^2+x(2)^2;
df=@(x) [0.02*x(1);2*x(2)];
ddf=@(x) [0.02 0;0 2];