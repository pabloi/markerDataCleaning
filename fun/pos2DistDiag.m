function [D,g,h]=pos2DistDiag(x,y)
    %Same as pos2Dist, but requires 2 inputs of equal size and assumes we only care about
    %the diagonal comparisons
    %x is Nxdim
    %y is Nxdim
    %D is Nx1 matrix containing distances
    %g is Nxdim containing gradient wrt X
    %h is N x Nxdim x Nxdim containing hessian wrt X 
    [N,dim]=size(x);
    if nargin<2 || isempty(y) || size(y,1)~=size(x,1)
        error('x and y should be same size')
    end
    gx=x-y; %Nxdim
    D=sqrt(sum(gx.^2,2)); %Nx1
    D1=D+eps; %Nx1
    if nargout>1 %Computing gradients too
        g=gx./D1;
        if nargout>2
            %TODO: vectorize hessian computation as is gradient, to avoid
            %for loops
                h=zeros(N,N,dim,N,dim);
%                 for j=1:N
%                     for k=1:dim
%                         h(j,:,j,k,j,k)=1; %Any way to make this assignment easier?
%                         h(:,j,j,k,j,k)=1;
%                         for i=1:N
%                             h(i,j,j,k,i,k)=-1;
%                             h(i,j,i,k,j,k)=-1;
%                         end
%                     end
%                 end
%                 h=h-(g.*reshape(g,M,N,1,1,N,dim));
%                 h=h./D;
%                 for j=1:N
%                     h(j,j,:,:,:,:)=0;
%                 end
        end
    end
end
