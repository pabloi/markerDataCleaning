function [D,g,h]=pos2Dist(x,y)
    %x is Nxdim
    %y is Mxdim
    %D is MxN matrix containing distances
    %g is MxN x Nxdim containing gradient wrt X
    %h is MxN x Nxdim x Nxdim containing hessian wrt X 
    singleInput=false;
    [N,dim]=size(x);
    if nargin<2 || isempty(y)
        y=x;
        singleInput=true;
        M=N;
    else
        [M]=size(y,1);
    end
    gx=reshape(x,1,N,dim)-reshape(y,M,1,dim); %MxNxdim
    D=sqrt(sum(gx.^2,3)); %MxN
    D1=D+eps; %MxN
    if nargout>1 %Computing gradients too
        g=zeros(M,N*N,dim);
        g(:,1:N+1:N*N,:)=gx./D1;
        g=reshape(g,M,N,N,dim);
        if singleInput
            g=g+permute(g,[2,1,3,4]); %Permute is very expensive, can it be avoided?
        end
        if nargout>2
            %TODO: vectorize hessian computation as is gradient, to avoid
            %for loops
            if ~singleInput
                h=zeros(M,N,N,dim,N,dim);
                for i=1:N
                    for k=1:dim
                        h(:,i,i,k,i,k)=1; %Any way to make this assignment easier?
                    end
                end
                h=h-(g.*reshape(g,M,N,1,1,N,dim));
                h=h./D;
            else %distances of x with respect to x, the gradient and hessian are more complicated
                h=zeros(M,N,N,dim,N,dim);
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
                h=h-(g.*reshape(g,M,N,1,1,N,dim));
                h=h./D;
                for j=1:N
                    h(j,j,:,:,:,:)=0;
                end
            end
        end
    end
end
