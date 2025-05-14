function yy=wmdeepyy(mm,zb,ranges,xx)
% Compute the output of fuzzy system with zb and ranges for input xx.
exapp=xx;
[numSamples,m]=size(exapp);
numInput=m;
for i=1:numInput
    fnCounts(i)=mm;
end;
activFns = zeros(numInput,2);
activGrades = zeros(numInput,2); 
baseCount(1)=1;
for i=2:numInput
    baseCount(i)=1;
    for j=2:i
        baseCount(i)=baseCount(i)*fnCounts(numInput-j+2);
    end;
end;
for j=1:numInput
    for i1=1:2^j
        for i2=1:2^(numInput-j)
            ma(i2+(i1-1)*2^(numInput-j),j)=mod(i1-1,2)+1;
        end;
    end;
end;
e1sum=0; 
for k = 1:numSamples
    for i = 1:numInput
        numFns = fnCounts(i);
        nthActive = 1;
        for nthFn = 1:numFns
            grade = meb2(numFns,nthFn,exapp(k,i),ranges(i,1),ranges(i,2));
            if grade > 0
                activFns(i,nthActive) = nthFn;
                activGrades(i,nthActive) = grade;
                nthActive = nthActive + 1;
            end;
        end;
    end; 
    for i=1:numInput
        nn(i,1)=activFns(i,1);
        nn(i,2)=nn(i,1)+1;
        if nn(i,1) == fnCounts(i)
            nn(i,2)=nn(i,1);
        end;
    end;
    a=0;
    b=0;
    for i=1:2^numInput
        indexcell=1;
        grade=1;
        for j=1:numInput
            grade=grade*activGrades(j,ma(i,j));
            indexcell=indexcell+(nn(numInput-j+1,ma(i,numInput-j+1))-1)*baseCount(j);
        end;
        a=a+zb(indexcell)*grade;
        b=b+grade;
    end;
    yy(k)= a/b; % the fuzzy system output
end; % endfor k
    

        
