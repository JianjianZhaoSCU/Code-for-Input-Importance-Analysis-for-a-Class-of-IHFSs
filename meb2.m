function y=meb2(n,i,x,xmin,xmax)
% Compute the value of the ith membership function in Fig. 2 at x.
h=(xmax-xmin)/(n-1);%每个三角形隶属度函数底边的一半
if i==1
    if x < xmin
        y=1;
    end;
    if x >= xmin & x < xmin+h
        y=(xmin-x+h)/h;
    end;
    if x >= xmin+h
        y=0;
    end;
end;
if i > 1 & i < n
    if x < xmin+(i-2)*h | x > xmin+i*h
        y=0;
    end;
    if x >= xmin+(i-2)*h & x < xmin+(i-1)*h
        y=(x-xmin-(i-2)*h)/h;
    end;
    if x >= xmin+(i-1)*h & x <= xmin+i*h
        y=(-x+xmin+i*h)/h;
    end;
end;
if i==n
    if x < xmax-h
        y=0;
    end;
    if x >= xmax-h & x < xmax
        y=(-xmax+x+h)/h;
    end;
    if x >= xmax
        y=1;
    end;
end;




