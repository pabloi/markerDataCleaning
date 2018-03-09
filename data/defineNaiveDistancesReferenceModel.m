load ./C0001MarkerData.mat %Training data and labels
%Fixing labels:
LH=strcmp(labels,'LHEE');
RH=strcmp(labels,'RHEE');
labels([find(LH),find(RH)])=labels([find(RH),find(LH)]); %Switch
model = naiveDistances(permute(M{8},[2,3,1]),labels);

%Define inequalities and bounds:
load naiveDistanceBounds.mat

%Save:
save ./distanceModelReferenceData.mat lowerBound upperBound model