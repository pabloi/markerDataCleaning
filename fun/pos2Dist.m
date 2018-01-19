function [D,g,h]=pos2Dist(x,y)
    %x is Nxdim
    %y is Mxdim
    %D is NxM matrix containing distances
    %g is NxM x Nxdim containing gradient wrt X
    %h is NxM x Nxdim x Nxdim containing hessian wrt X
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
    D1=D;
    D1(D==0)=1;
    if nargout>1 %Computing gradients too
        gx=xx./D1; %allowed in R2017a and newer
        g=zeros(N*N,M*dim); %Reshaping to NM x (N.dim) size. Should be sparse?
        g(1:N+1:N*N,:)=gx(:,:);
        g=reshape(g,N,N,M,dim);
        if singleInput
            g=g+permute(g,[3,2,1,4]); %Permute is very expensive, can it be avoided?
        end
        g=permute(g,[1,3,2,4]);
        if nargout>2
            %TODO: vectorize hessian computation as is gradient, to avoid
            %for loops
            if ~singleInput
                h=zeros(N,M,N,dim,N,dim);
                for i=1:N
                    for k=1:dim
                        h(i,:,i,k,i,k)=1; %Any way to make this assignment easier?
                    end
                end
                h=h-(g.*reshape(g,N,M,1,1,N,dim));
                h=h./D;
            else %distances of x with respect to x, the gradient and hessian are more complicated
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
end
