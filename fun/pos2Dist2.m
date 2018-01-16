function [D,g,h]=pos2Dist2(x,y)
    %x is Nxdim
    %y is Mxdim
    %D is NxM matrix containing distances
    %g is NM x N.dim containing gradient wrt X
    %h is NM x N.dim x N.dim containing hessian wrt X
    singleInput=false;
    if nargin<2 || isempty(y)
        y=x;
        singleInput=true;
    end
    [N,dim]=size(x);
    [M,dim]=size(y);
    xx=bsxfun(@minus,x,reshape(y',1,dim,M)); %N x dim x M
    xx=permute(xx,[1,3,2]); %NxMxdim
    D=(sum(xx.^2,3)); %NxM
    if nargout>1 %Computing gradients too
        gx=2*xx;
        g=zeros(N,M,N,dim); %Reshaping to NM x (N.dim) size. Should be sparse?
        h=zeros(N,M,N,dim,N,dim);
        for i=1:N
            g(i,:,i,:)=gx(i,:,:);
            for k=1:dim
                h(i,:,i,k,i,k)=2; %Any way to make this assignment easier?
            end
        end
        if singleInput %distances of x with respect to x, the gradient and hessian are more complicated
            h=zeros(N,M,N,dim,N,dim);
            for j=1:N
                g(j,j,:,:)=0; %Diagonal distances are constant=0
                for k=1:dim
                    h(j,:,j,k,j,k)=1; %Any way to make this assignment easier?
                    h(:,j,j,k,j,k)=1;
                    for i=1:N
                        h(i,j,j,k,i,k)=-1;
                        h(i,j,i,k,j,k)=-1;
                    end
                end
            end
            g=g+permute(g,[2,1,3,4]);
            for j=1:N
                h(j,j,:,:,:,:)=0;
            end
            h=2*h;
        end
        
        %gy=-gx; -> Gradients are opposite to one another if we preserve the shape
    end
end