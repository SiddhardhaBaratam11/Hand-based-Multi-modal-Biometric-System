
load('feat_knuckle_squeezenet.mat');

load('feat_vein_efficientnet.mat');
load('Labels.mat');
%features3=[features2,zeros(500,48896)];
%save feat_vein_new.mat features3;
%fused_feat = (0.9099*features1)+ (0.8582*features3);
fused_feat = features1 + features2;
save fused_feat_opt.mat  fused_feat;
