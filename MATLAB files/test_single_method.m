function x = test_single_method(G,f,df,ddf,testpts,startpt,a,b,res)
% G: optimization method to test; required signature: (f,df,ddf,x0,a,b,iter)
% f: R->R objective
% df: derivative
% ddf: 2nd derivative
% testpts: number of iterations for G
% startpt: starting point
% a,b: search and plot interval = [a,b]
% res: plot resolution = res
% x: output after each iteration

    X = linspace(a,b,res);
    Y = f(X);
    
    x = zeros(1,testpts);
    for k=1:testpts
        x(k) = G(f,df,ddf,startpt,a,b,k);
    end
    plot(X,Y,x,f(x),'o');
end
