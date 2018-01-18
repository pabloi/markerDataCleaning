function [D,g,h]=pos2Dist(x,y)
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
    xx=x-permute(y,[3,2,1]);
    xx=permute(xx,[1,3,2]); %NxMxdim
    D=sqrt(sum(xx.^2,3)); %NxM
    if nargout>1 %Computing gradients too
        %gx=bsxfun(@rdivide,xx,D); %NxMxdim
        gx=xx./D; %allowed in R2017a and newer
        for k=1:3
            aux=gx(:,:,k);
            aux(D==0)=1;
            gx(:,:,k)=aux;
        end
        g=zeros(N,M,N,dim); %Reshaping to NM x (N.dim) size. Should be sparse?
        for i=1:N
            g(i,:,i,:)=gx(i,:,:); %Any way to make this faster?
        end
        if nargout>2
            h=zeros(N,M,N,dim,N,dim);
            for i=1:N
                for k=1:dim
                    h(i,:,i,k,i,k)=1; %Any way to make this assignment easier?
                end
            end
            h=h-(g.*reshape(g,N,M,1,1,N,dim));
            h=h./D;
        end
        if singleInput %distances of x with respect to x, the gradient and hessian are more complicated
            for j=1:N
                g(j,j,:,:)=0; %Diagonal distances are constant=0
            end
            g=g+permute(g,[2,1,3,4]);
            if nargout>2
                h=zeros(N,M,N,dim,N,dim);
                for j=1:N
                for k=1:dim
                    h(j,:,j,k,j,k)=1; %Any way to make this assignment easier?
                    h(:,j,j,k,j,k)=1;
                    for i=1:N
                        h(i,j,j,k,i,k)=-1;
                        h(i,j,i,k,j,k)=-1;
                    end
                end
                end
                h=h-(g.*reshape(g,N,M,1,1,N,dim));
                h=h./D;
                for j=1:N
                    h(j,j,:,:,:,:)=0;
                end
            end
        end
    end
end