function [altPos] = getBestReconsFromZeroPlusModel(measuredPos,D,sD,meanPos,I,biasPos)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
altPos=nan(size(measuredPos));
if nargin<6 || isempty(biasPos)
    biasPos=zeros(size(measuredPos));
end
for i=1:size(measuredPos,1) %Only reconstructing actual markers
    X1=measuredPos;
    X1(i,:)=nan;
    [X2] = expandDataForZeroPlus(X1,I);
    [altPos(i,:)] = getBestReconsFromZeroModel(X2,D,sD,meanPos,i,biasPos); %Only reconstructing i-th marker
end
end

