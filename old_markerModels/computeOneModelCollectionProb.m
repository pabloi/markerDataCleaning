function [p] = computeOneModelCollectionProb(actualX,D,sD)

actualD=computeDistanceMatrix(actualX);
[N,N1,M1]=size(actualD);
[N,~,M]=size(D);

V1=reshape(actualD,N^2,M1);
V2=reshape(D,1,N^2,M);
auxMat=bsxfun(@minus,V1',V2).^2;
clear V1 V2
dist=reshape((nanmean(auxMat,2)),M1,M); %Squared distances
clear auxMat
p=bsxfun(@rdivide,1./dist,nansum(1./dist,2));

end

