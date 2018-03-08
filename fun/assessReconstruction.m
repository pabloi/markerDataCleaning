function assessReconstruction(referenceData, reconstructedData, model)

%% Assess missing and outlier markers in original & reconstructed data
missing=squeeze(any(isnan(referenceData),2));
outlier=model.outlierDetect(referenceData);
missingR=squeeze(any(isnan(reconstructedData),2));
outlierR=model.outlierDetect(reconstructedData);
disp(['Original missing markers: ' num2str(sum(missing(:)))])
disp(['Reconstructed missing markers: ' num2str(sum(missingR(:)))])
disp(['Original outlier markers: ' num2str(sum(outlier(:)))])
disp(['Reconstructed outlier markers: ' num2str(sum(outlierR(:)))])
%% Assess distortion from present but not outlier markers during reconstruction
dist=reconstructedData-referenceData;
figure;
for i=1:3
subplot(2,3,i)
histogram(squeeze(dist(:,i,:))')
title('X'+(i-1))
end
for i=1:3
subplot(2,3,i+3)
plot(squeeze(dist(:,i,:))')
end

%% Assess likelihood differences between original and reconstructed data
%marker level  (could do it at distance level)
L=model.scoreMarkersOpt(referenceData);
LR=model.scoreMarkersOpt(reconstructedData);
figure;
for i=1:size(L,1)
subplot(5,5,i)
histogram(L(i,:),[-Inf,-10:.1:0])
hold on
histogram(LR(i,:),[-Inf,-10:.1:0])
title(model.markerLabels{i})
if i==1
legend('Original','Reconstructed')
end
end
subplot(5,5,21:25)
title('Frame log-log-likelihoods')
semilogy((-max(-L)))
hold on
semilogy((-max(-LR)))
legend('Original','Reconstructed')

end