clc; clear all; close all;

imds2 = imageDatastore('D:\education\notes\sem-4\mmp project\knuckle\exp\Database\knuckle',... %(After splitting)
     'IncludeSubfolders',true,...
     'LabelSource','foldernames');
 inputSize = [227 227 3];

imds_aug2=augmentedImageDatastore(inputSize, imds2,'ColorPreprocessing','gray2rgb');



%% Load a pre-trained, deep, convolutional network
net = squeezenet();
%% Modify the network to use four categories

layer='conv10';
%   layers=[imageInputLayer([200 200 1])
%      net(2:end-3)
%      fullyConnectedLayer(2)
%      softmaxLayer
%       classificationLayer
%        ];
%   
% opts1 = trainingOptions('sgdm', 'InitialLearnRate', 0.001, ... 
%        'MaxEpochs',7, 'MiniBatchSize', 100, ...  
%        'ExecutionEnvironment', 'cpu', ...
%        'Shuffle','every-epoch', ...
%        'Plots','training-progress');
%  myNet2 = trainNetwork(imds_aug2,layers);

% save('mdl_Alex_xray', 'myNet2');
% load('myNet2');
features1 = activations(net,imds_aug2,layer,'OutputAs','rows','ExecutionEnvironment','cpu');
% feat_1=features1(:,1);
labels1 = imds2.Labels;
save feat_knuckle_squeezenet features1;
% Measure network accuracy 
% train_net = fitcecoc(features2(1:400,:), imds2.Labels(1:400,:),'Learners','svm');
% predictedLabels1 = predict(train_net,features2(401:500,:)); % testing error
%  accuracy = mean(predictedLabels1  == imds2.Labels(401:500,:))
% confusionchart(imds2.Labels(401:500,:),predictedLabels1)
%%%% VISUALIZE THE FEATUREMAP IN Squeezenet %%%%%%%z
layer = 4;
 name = net.Layers(layer).Name;
 channels = 1:36;
 I = deepDreamImage(net,name,channels, ...
     'PyramidLevels',1);
figure
 I = imtile(I,'ThumbnailSize',[64 64]);
 imshow(rgb2gray(I))
 title(['Layer ',name,' Features'],'Interpreter','none')


 %%%%Measure network accuracy for XRAY DB 
 train_net = fitcecoc(features1(1:2:500,:), imds2.Labels(1:2:500,:),'Learners','svm');
 predictedLabels1 = predict(train_net,features1(2:2:500,:)); % testing error
  accuracy = mean(predictedLabels1  == imds2.Labels(2:2:500,:));
 confusionchart(imds2.Labels(2:2:500,:),predictedLabels1)