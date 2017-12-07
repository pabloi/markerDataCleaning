%%
%Idea: given some candidate outliers scored with integers based on how many
%outlier stats each marker participates, find a classification (binary) for
%the minimum number of outlier markers that explains the outlier stats.
%Only works if stats are always computed from 2 and only 2 markers.
%Let p be the outlier scores of each markers' stats, then the problem is a
%mixed-integer programming of the form:
%min 1'*x [minimum number of outlier markers possible]
%subject to 
%y+z = p - 1'*x; y>=0; z<=0; [y quantifies how many more outlier stats
%than there are outlier markers each marker has; z how many less, one of
%the two is always 0]
%y>=x & y-max(p)*x<=0 [these two enforce x=(y>0)]

%%
beq=[1 1 4 2 3 0 1]';
M=numel(beq);
uno=ones(M,1);
cero=zeros(M,1);

%Strict equality: z+y = b-1'*x -> b = 1'*x+y+z;
Aeq=[ones(M) eye(M) eye(M)]; %M x 3M

%Inequality: y>=x -> x-y <= 0
A=[eye(M), -eye(M), zeros(M)];
b=cero;
%Inequality: y-max(b)*x <=0 
A=[A; -max(beq)*eye(M), eye(M), zeros(M)];
b=[b;cero];

%Constraints:
lb=[cero;cero;-Inf*uno];
ub=[uno; Inf*uno;cero];

%% Solve
f=[uno;cero;cero];
xy=intlinprog(f,1:M,A,b,Aeq,beq,lb,ub); %Solving for integer (binary) x
x=xy(1:M)
y=xy(M+1:2*M);
z=xy(2*M+1:end);