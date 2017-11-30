function [D,sD,meanPos,I,biasPos] = createZeroPlusModel(inputData1)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

if isa(inputData1,'char') %Assume it is the name of a .c3d file
    %Load c3d using btk, to be done
elseif isa(inputData1,'orientedLabTimeSeries') %Assume it is a timeseries containing 3D data
    X=inputData.getOrientedData;
else %Assume it is a Nx3xM matrix
   X=inputData1; 
end

%First step: increas the inputData matrix by adding virtual markers
[inputData,I,sourceI] = expandDataForZeroPlus(inputData1);

%Second step: do a zero model
[D,sD,meanPos] = createZeroModel(inputData);

%Third step: from the zero model, remove the cross-terms that involve a
%marker and virtual marker generated from itself
tol=1e3;
[N]=size(inputData1,1);
for k=1:size(sourceI,1)
    sD(N+k,sourceI(k,:))=tol;
    sD(sourceI(k,:),N+k)=tol;
end

%Fourth step: learn a bias for the model
X1a=inputData1;
Xbar=nan(size(X1a));
for i=1:size(X1a,3)
   Xbar(:,:,i)=getBestReconsFromZeroPlusModel(X1a(:,:,i),D,sD,meanPos,I); 
end
dd=squeeze(Xbar-X1a);
biasPos=mean(dd,3);
end

