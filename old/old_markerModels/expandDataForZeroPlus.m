function [newData,I,sourceI] = expandDataForZeroPlus(posData,I)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here


[N,dim,M]=size(posData);

%These virtual markers don't improve anything:
% virtualData=bsxfun(@minus,2*posData,reshape(permute(posData,[2,3,1]),[1,dim,M,N]));
% virtualData=permute(virtualData,[1,4,2,3]);
% virtualData=reshape(virtualData,N^2,dim,M);
% sourceI(:,1)=repmat(1:N,N,1);
% sourceI(:,2)=repmat([1:N]',1,N);
 

%Virtual markers from cross-products:
iii=nchoosek(1:N,3);
P=size(iii,1);
virtualData=nan(P*3,dim,M);
for i=1:P
    for k=1:3
        switch k
            case 1
                k1=2;
                k2=3;
            case 2
                k1=1;
                k2=3;
            case 3
                k1=2;
                k2=3;
        end
        vec1=posData(iii(i,k1),:,:)-posData(iii(i,k),:,:);
        nVec1=sqrt(sum(vec1.^2,2));
        vec1=bsxfun(@rdivide,vec1,nVec1);
        vec2=posData(iii(i,k2),:,:)-posData(iii(i,k),:,:);
        nVec2=sqrt(sum(vec2.^2,2));
        vec2=bsxfun(@rdivide,vec2,nVec2);
        aux=cross(vec1,vec2);
        virtualData(3*(i-1)+k,:,:)=bsxfun(@times,aux,.5*(nVec1+nVec2))+posData(iii(i,k),:,:);
    end
end
sourceI(1:3:P*3,:)=iii;
sourceI(2:3:P*3,:)=iii;
sourceI(3:3:P*3,:)=iii;


if nargin<2 || isempty(I)
    %Dropping some virtual markers for complexity's sake:
    [D,sD,meanPos] = createZeroModel(posData);
    tol=15;
    I=sD(sub2ind(size(sD),sourceI(:,1),sourceI(:,2)))<tol & sD(sub2ind(size(sD),sourceI(:,1),sourceI(:,2)))>0;
    if size(sourceI,2)>2
        I2=sD(sub2ind(size(sD),sourceI(:,1),sourceI(:,3)))<tol & sD(sub2ind(size(sD),sourceI(:,1),sourceI(:,3)))>0;
        I3=sD(sub2ind(size(sD),sourceI(:,2),sourceI(:,3)))<tol & sD(sub2ind(size(sD),sourceI(:,2),sourceI(:,3)))>0;
        I = I & I2 & I3;
    end
    
    %For the pair-based markers:
%     %Distance based:
%     D=computeDistanceMatrix(nanmean(posData,3));
%     I1=D<.8*nanmedian(D(:)) & D>0; %Only marker-pairs below 40% of median distances will be considered
%     
%     %std of distance based: this doesn't work because all virtual markers
%     %satisfy the condition
%     [D,sD1,meanPos] = createZeroModel(cat(1,posData,virtualData));
%     sD=sD1(N+1:end,:);
%     I=any(sD>0 & sD<10,2); %Only variability in distance of less than 10mm is preserved
%     I=reshape(I,[N,N]);
%     
%     Another attempt: based on variability between the two involved markers
%     I=sD>0 & sD<10; %Only variability in distance of less than 10mm is preserved
%     I=I(:);
    
end

sourceI=sourceI(I==1,:);
newData=cat(1,posData,virtualData(I==1,:,:));

end

