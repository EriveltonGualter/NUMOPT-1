function x = zero_unimodal( f,~,~,~,a,b,iter )
% f: real->real objective function
% [a,b]: interval
% iter: number of iterations
% x: estimation of minimizer

    ak=a; bk=b;
    for k=1:iter
        xl=(2*ak+bk)/3;
        xh=(ak+2*bk)/3;
        if f(xl)<f(xh)
            bk=xh;
        else
            ak=xl;
        end
    end
    x=(ak+bk)/2;
end
