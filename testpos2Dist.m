%% Data
N=4;
dim=3;
X=randn(N,dim);
M=2;
Y=randn(M,dim);
%% comparing gradient in cross-distance to empirical results
%[d,g,h]=pos2Dist2(X,Y);
[d,g,h]=pos2Dist(X,Y);
epsilon=1e-7;
empG=nan(size(g));
empH=nan(size(h));
for i=1:N
    for k=1:dim
        aux=zeros(size(X));
        aux(i,k)=epsilon;
        %[d1,g1,h1]=pos2Dist2(X+aux,Y);
        [d1,g1,h1]=pos2Dist(X+aux,Y);
        empG(:,:,i,k)=(d1-d)/epsilon;
        empH(:,:,:,:,i,k)=(g1-g)/epsilon;
    end
end
disp(['Max gradient element: ' num2str(max(abs(g(:))))])
disp(['Max gradient err: ' num2str(max(abs(g(:)-empG(:))))])
disp(['Max gradient err (%): ' num2str(100*max(abs(g(:)-empG(:))./abs(g(:))))])

disp(['Max hessian element: ' num2str(max(abs(h(:))))])
disp(['Max hessian err: ' num2str(max(abs(h(:)-empH(:))))])
disp(['Max hessian err (%): ' num2str(100*max(abs(h(:)-empH(:))./abs(h(:))))])

%% comparing gradient in self-distance to empirical results
[d,g,h]=pos2Dist(X);
%[d,g,h]=pos2Dist2(X);
epsilon=1e-7;
empG=nan(size(g));
empH=nan(size(h));
for i=1:N
    for k=1:dim
        aux=zeros(size(X));
        aux(i,k)=epsilon;
        [d1,g1,h1]=pos2Dist(X+aux);
        %[d1,g1,h1]=pos2Dist2(X+aux);
        empG(:,:,i,k)=(d1-d)/epsilon;
        empH(:,:,:,:,i,k)=(g1-g)/epsilon;

    end
end
disp(['Max gradient element: ' num2str(max(abs(g(:))))])
disp(['Max gradient err: ' num2str(max(abs(g(:)-empG(:))))])
disp(['Max gradient err (%): ' num2str(100*max(abs(g(:)-empG(:))./abs(g(:))))])

disp(['Max hessian element: ' num2str(max(abs(h(:))))])
disp(['Max hessian err: ' num2str(max(abs(h(:)-empH(:))))])
disp(['Max hessian err (%): ' num2str(100*max(abs(h(:)-empH(:))./abs(h(:))))])