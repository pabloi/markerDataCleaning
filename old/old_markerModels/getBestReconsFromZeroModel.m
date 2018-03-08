function [altPos] = getBestReconsFromZeroModel(measuredPos,D,sD,meanPos,markersForReconstruction,biasPos)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
X2=measuredPos;
[~,~,x0]=getRotationAndTranslation(meanPos(1:size(measuredPos,1),:),measuredPos);

if nargin<5 || isempty(markersForReconstruction) %If marker list not given, reconstructing everything
    markersForReconstruction=1:size(measuredPos,1);
end
X2alt=X2;
if nargin<6 || isempty(biasPos)
    biasPos=zeros(size(measuredPos));
end

measuredLike=determineLikelihoodFromZeroModel(X2alt,D,sD);
measuredLike(isnan(measuredLike))=-1e3;
[~,markerOrder]=sort(measuredLike(markersForReconstruction),1,'ascend'); %Sorting likelihoods of measuredMarkers that need to be reconstructed
lp=determineLikelihoodFromZeroModel(measuredPos,D,sD);

kP=X2;
kD=D(:,markersForReconstruction);
sD2=sD+diag(nan*diag(sD));
sD=sqrt(sD.^2+0); %Adding some uncertainty for measurement errors (?)
selfSD=min(sD2,[],1).*exp((lp+1));
%Weight for same-marker will be inversely related to likelihood of measurement. Likelihood ~-1 weights measurement as the best reference marker, -2 weights it 1/e, -3 weights it 1/e^2
selfSD=1e3*ones(size(selfSD))
kSD=sD+diag(selfSD); 
kSD=kSD(:,markersForReconstruction);
    
for i=1:length(markerOrder)
    marker=markersForReconstruction(markerOrder(i)); %Marker to reconstruct
    
    w=1./kSD(:,marker); 
    w=w/nansum(w);
    wS=sort(w,'descend');
    
    idx=any(isnan(kP),2) | isnan(kD(:,marker)) | isnan(w) | (w<(1/length(w)) & w<wS(4)); %Eliminating missing markers & very-low weights (preserving at least 4 reference markers)
    [X2alt(marker,:)] = getPositionFromDistances(kP(~idx,:),kD(~idx,marker),w(~idx),x0(marker,:)); %Estimated reconstruction
    kP=X2alt; %Update
end
altPos=X2alt+biasPos;

end

