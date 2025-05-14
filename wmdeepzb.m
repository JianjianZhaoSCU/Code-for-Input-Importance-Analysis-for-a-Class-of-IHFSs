function [zb ranges]=wmdeepzb(mm,xx,y)
% Train fuzzy system with input xx and output y to get zb and ranges,
% where zb is the c^{j_{1},...,j_{m}}in FS (7) and ranges is the endpoints in (11).
extra=[xx,y]; %extra=[input,output]
[numSamples,m]=size(extra); %numSamples=样本个数，m=输入+输出的样本维度
numInput=m-1;%numInput=输入样本维度
for i=1:numInput
    fnCounts(i)=mm; %mm为每个输入的模糊集个数
end;
ranges = zeros(numInput,2); %ranges为每个输入的取值范围的端点值
activFns = zeros(numInput,2);%activFns为输入变量激活的隶属度函数标号
activGrades = zeros(numInput,2); %activGrades为输入变量激活的隶属度函数值
searchPath = zeros(numInput,2); %searchPath为
numCells = 1; % number of regions (cells)为输入空间的格子数
grade_count = zeros(numInput,1); %记录每个变量有多少个隶属度函数等于1的值
for i = 1:numInput
    ranges(i,1) = min(extra(:,i));
    ranges(i,2) = max(extra(:,i));
    numCells = numCells * fnCounts(i);
end;
baseCount(1)=1;
for i=2:numInput
    baseCount(i)=1;
    for j=2:i
        baseCount(i)=baseCount(i)*fnCounts(numInput-j+2);
    end;
end;
% Generate rules for cells covered by data 
zb = zeros(1,numCells); % THEN part centers of generated rules
ym = zeros(1,numCells);
for k = 1:numSamples
    for i = 1:numInput
        numFns = fnCounts(i);%numFns为每个输入的隶属度函数个数
        nthActive = 1;
        for nthFn = 1:numFns %nthFns代表第几个隶属度函数
            grade = meb2(numFns,nthFn,extra(k,i),ranges(i,1),ranges(i,2));
            if grade > 0
                activFns(i,nthActive) = nthFn;
                activGrades(i,nthActive) = grade;
                nthActive = nthActive + 1;
                if grade == 1
                    grade_count(i,1) = grade_count(i,1) + 1;
                end;
            end;
        end;%endfor nthFn
    end;%endfor i
    for i=1:numInput
        if activGrades(i,1) >= activGrades(i,2)
            searchPath(i,1)=activFns(i,1);
            searchPath(i,2)=activGrades(i,1);
        else
            searchPath(i,1)=activFns(i,2);
            searchPath(i,2)=activGrades(i,2);
        end;
    end;
    indexcell=1;%格子索引（第几条规则）
    grade=1;%该格子（规则）的激活强度
    for i=1:numInput
        grade=grade*searchPath(i,2);
        indexcell=indexcell+(searchPath(numInput-i+1,1)-1)*baseCount(i);
    end;
    ym(indexcell)=ym(indexcell)+grade;%式（13）的w
    zb(indexcell)=zb(indexcell)+extra(k,numInput+1)*grade;%式（14）的u
end; % endfor k
for j=1:numCells
    if ym(j) ~= 0
        zb(j)=zb(j)/ym(j);
    end;
end;
% Extrapolate the rules to all the cells 
for i=1:numInput-1
    baseCount(i)=1;
    for j=1:i
        baseCount(i)=baseCount(i)*fnCounts(numInput-j+1);
    end;
end;
ct=1;
zbb = zeros(1,numCells);
ymm = zeros(1,numCells);
while ct > 0
%只要 ct（计数器）大于零，循环就会继续运行。
    ct=0;
    %统计 ym 中的零值
    for s=1:numCells
        if ym(s) == 0
            ct=ct+1;
        end;
    end;
    %处理 ym 中值为零的元素
    for s=1:numCells
        if ym(s) == 0
            s1=s;
            index = ones(1,numInput);
            %将 s 转换为多维索引index
            for i=numInput-1:-1:1
                while s1 > baseCount(i)
                    s1=s1-baseCount(i);
                    index(numInput-i)=index(numInput-i)+1;
                end;
            end;
            index(numInput)=s1;
            zbnum=0;%初始化邻居加权累积变量
            for i=1:numInput-1 %遍历前 numInput-1 维，累积邻居的值：
                if index(i) > 1 %如果 index(i) > 1，当前单元存在前向邻居（左）
                    %将邻居的 zb 和 ym 加入累积变量 zbb(s) 和 ymm(s)。
                    zbb(s)=zbb(s)+zb(s-baseCount(numInput-i));
                    ymm(s)=ymm(s)+ym(s-baseCount(numInput-i));
                    %增加计数 zbnum，仅统计 ym 不为零的邻居。
                    zbnum=zbnum+sign(ym(s-baseCount(numInput-i)));
                end;
                if index(i) < fnCounts(i) %如果 index(i) < fnCounts(i)，当前单元存在后向邻居（右）
                    zbb(s)=zbb(s)+zb(s+baseCount(numInput-i));
                    ymm(s)=ymm(s)+ym(s+baseCount(numInput-i));
                    zbnum=zbnum+sign(ym(s+baseCount(numInput-i)));
                end;
            end;
            if index(numInput) > 1 %当前单元存在下向邻居
                zbb(s)=zbb(s)+zb(s-1);
                ymm(s)=ymm(s)+ym(s-1);
                zbnum=zbnum+sign(ym(s-1));
            end;
            if index(numInput) < fnCounts(numInput) %当前单元存在上向邻居
                zbb(s)=zbb(s)+zb(s+1);
                ymm(s)=ymm(s)+ym(s+1);
                zbnum=zbnum+sign(ym(s+1));
            end;
            if zbnum >= 1
                zbb(s)=zbb(s)/zbnum;
                ymm(s)=ymm(s)/zbnum;
            end;
        end; % endif ym   
    end; % endfor s
    for s=1:numCells
        if ym(s) == 0 & ymm(s) ~= 0
            zb(s)=zbb(s);
            ym(s)=ymm(s);
        end;
    end;
end; % endwhile ct