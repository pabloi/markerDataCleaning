% Test distance model:

%% Load data
clearvars

load ./data/LI16_Trial9_expData.mat %processedTrialData
data=LI16_Trial9_expData.markerData;

%% Build model
mm=data.buildNaiveDistancesModel;

%%
[~,~,data]=data.assessMissing([],-1);
data=data.findOutliers(mm);
