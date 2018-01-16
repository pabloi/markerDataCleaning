function [pos] = getPositionFromDistances_v3(knownPositions,knownDistances,posWeights,distWeights,initGuess)
%v3: many changes
%pos is the Nx3 matrix that minimizes:
%sum((posWeights).*(pos-knownPositions)*(pos-knownPositions)')+ .5*sum(sum((distWeights.*abs(d(pos_i,pos_j)-knownDistances)))
%INPUT:
%knownPositions: N x D  matrix, D being dimension of space
%knownDistances: NxN matrix
%posWeights: Nx1 weight vector
%distWeights: NxN weight matrix
%OUTPUT:
%pos: N x D matrix containing D-dimensional positions for N points

%TODO: if posWeights=Inf for some component, it gets excluded from the
%optimization. If posWeights=0, it gets excluded from knownPositions.

[N,dim]=size(knownPositions);
[N1,N2]=size(knownDistances);
if N~=N1 || N1~=N2
    error('Dimension mismatch')
end
if nargin<3 || isempty(posWeights)
   posWeights= ones(size(knownPositions));
elseif size(posWeights,1)~=N
    error('Weight dimensions mismatch')
end
if nargin<4 || isempty(distWeights)
    distWeights=ones(size(knownDistances));
elseif size(distWeights,1)~=N || size(distWeights,2)~=N
    error('Weight dimensions mismatch')
end

if nargin<5 || isempty(initGuess)
    initGuess=mean(knownPositions);
end

if N1~=N
    error('Provided distances dimension mismatch. Check that the number of distances is the same as the numer of known positions')
end
distWeights=triu(distWeights,1); %Because distances are doubled, I am only honoring the upper half of the distribution

%Option 1:
%Do a least-squares regression:
opts = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,'HessianFcn','objective','Display','off');
pos=fminunc(@(x) cost(reshape(x,N,dim),knownPositions,knownDistances,posWeights,distWeights),initGuess(:),opts);

pos=reshape(pos,N,dim);
end

function [f,g,h]=cost(x,kP,kD,wP,wD)
    [M,dim]=size(x);
    [N,dim]=size(kP);
    [D1,g1,h1]=pos2Dist(x);  %Can also use pos2Dist2 for quadratic weighing   
    [D2,g2,h2]=pos2Dist(x,kP); %We care only about the diagonal of this
    f=sum(sum(wD.*abs(D1-kD))) + sum(sum(diag(wP).*D2,1),2);
    g=reshape(sum(sum(wD.*sign(D1-kD).*g1,2),1),N*dim,1) + reshape(sum(sum(diag(wP).*g2,1),2),N*dim,1);
    h=reshape(sum(sum(wD.*sign(D1-kD).*h1,2),1),N*dim,N*dim) + reshape(sum(sum(diag(wP).*h2,1),2),N*dim,N*dim);
end

%% A little script to test cost:
% %% Data
% N=4;
% dim=3;
% X=randn(N,dim);
% x2=randn(N,dim);
% kD=randn(N,N);
% wP=randn(N,1);
% wD=randn(size(kD));
% 
% %% comparing gradient in cross-distance to empirical results
% 
% [d,g,h]=cost(X,x2,kD,wP,wD);
% epsilon=1e-7;
% empG=nan(N,dim);
% empH=nan(N,dim,N,dim);
% for i=1:N
%     for k=1:dim
%         aux=zeros(size(X));
%         aux(i,k)=epsilon;
%         [d1,g1,h1]=cost(X+aux,x2,kD,wP,wD);
%         empG(i,k)=(d1-d)/epsilon;
%         empH(:,:,i,k)=(reshape(g1,N,dim)-reshape(g,N,dim))/epsilon;
%     end
% end
% disp(['Max gradient element: ' num2str(max(abs(g(:))))])
% disp(['Max gradient err: ' num2str(max(abs(g(:)-empG(:))))])
% disp(['Max gradient err (%): ' num2str(100*max(abs(g(:)-empG(:))./abs(g(:))))])
% 
% disp(['Max hessian element: ' num2str(max(abs(h(:))))])
% disp(['Max hessian err: ' num2str(max(abs(h(:)-empH(:))))])
% disp(['Max hessian err (%): ' num2str(100*max(abs(h(:)-empH(:))./abs(h(:))))])
