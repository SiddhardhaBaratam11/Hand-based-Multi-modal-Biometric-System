function [z] = error_rate1(kernel_pars)
    % load P; load TV; load TVT; load T;
alpa=kernel_pars(1);
beta=kernel_pars(2);
% % gamma=kernel_pars(3);
load('feat_knuckle_squeezenet.mat')

load('Labels.mat');
load('feat_vein_efficientnet.mat');

%load('new_label_final_vggnet.mat')
% alpha=1;beta=1;
% 
% %%%%%PERFORMANCE EVALUATION FOR 70 % TRAINING USING SVM %%%%%
% 
feat_vein_new_train=[alpa*features1+ beta*features2];
%feat_vein_new_train=org_feat;
%Labels=org_label;
train_net = fitcecoc(feat_vein_new_train, Labels,'Learners','svm');

predictedLabels1 = predict(train_net,feat_vein_new_train); % testing error
 accuracy = mean(predictedLabels1  == Labels);
% confusionchart(new_label_final_alexnet(2:2:6400,:),predictedLabels1)
% plotconfusion(label_test,predictedLabels1)


   
% [out_class] = main_1d_CNN_classifier(conc_feat,label,conc_feat,label);
% [C, order] = confusionmat(label', out_class'); 
% [acc,frr,far]=cal_far_frra_acc(C);
% Exact solutions should be (1,1,...,1) 
z=1-accuracy;