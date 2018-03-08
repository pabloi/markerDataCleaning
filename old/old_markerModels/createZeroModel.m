function [D,sD,meanPos,biasPos] = createZeroModel(inputData)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

if isa(inputData,'char') %Assume it is the name of a .c3d file
    %Load c3d using btk, to be done
elseif isa(inputData,'orientedLabTimeSeries') %Assume it is a timeseries containing 3D data
    X=inputData.getOrientedData;
else %Assume it is a Nx3xM matrix
   X=inputData; 
end

[N,dim,M]=size(X);
dist=nan(N,N,M);
for i=1:M
    dist(:,:,i)=computeDistanceMatrix(X(:,:,i));
end



%D=trimmean(dist,10,'round',3); %Mean across 3rd dim, trimming 10% of extreme values
D=nanmedian(dist,3);
sD=1.4826*mad(dist,1,3); %Computes the median absolute deviation, as a more robust estimation of std() when outliers are present. 1.48 is the factor between sigma and mad in a normal dist


meanPos=nanmedian(inputData,3);

if nargout>3
    altPos=nan(size(inputData));
    for i=1:size(inputData,3)
        [altPos(:,:,i)] = getBestReconsFromZeroModel(inputData(:,:,i),D,sD,meanPos);
    end
    biasPos=nanmean(altPos-inputData,3);
else
    biasPos=zeros(N,dim);
end

end

