clc; clear all; close all;

imds1 = imageDatastore('D:\education\notes\sem-4\mmp project\knuckle\exp\Database\vein',... %(After splitting)
     'IncludeSubfolders',true,...
     'LabelSource','foldernames');
% imageAugmenter = imageDataAugmenter( ...)
 inputSize = [224 224 3];

imds_aug1=augmentedImageDatastore(inputSize, imds1,'ColorPreprocessing','gray2rgb');



%% Load a pre-trained, deep, convolutional network
net = efficientnetb0;
%% Modify the network to use four categories
% 
 layer='efficientnet-b0|model|head|global_average_pooling2d|GlobAvgPool';

lgraph = layerGraph(net);
numClasses = 100;
newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'fc_final')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')];
lgraph = removeLayers(lgraph, {'efficientnet-b0|model|head|dense|MatMul', 'Softmax', 'classification'});
lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'efficientnet-b0|model|head|global_average_pooling2d|GlobAvgPool', 'fc_final');

 opt=trainingOptions("sgdm",'MaxEpochs',5,....
     'InitialLearnRate',0.0001,'Plots','training-progress');

myNet1 = trainNetwork(imds_aug1,lgraph,opt);

save('mdl_efficientnet_tuned_vein', 'myNet1');
% load('myNet1');
features2 = activations(myNet1,imds_aug1,layer,'OutputAs','rows','ExecutionEnvironment','cpu');
% feat_1=features2(:,1);
labels1 = imds1.Labels;
save feat_vein_efficientnet.mat features2;
% Measure network accuracy for XRAY DB 
Model=fitcecoc(features2,labels1);
predict_1=predict(Model,features2);
acc=mean(labels1==predict_1);
confusionchart(labels1,predict_1)
plotconfusion(labels1,predict_1)
%%%% VISUALIZE THE FEATUREMAP IN GoogLenet %%%%%%%%
layer = 288;
 name = net.Layers(layer).Name;
 channels = 1:36;
 I = deepDreamImage(net,name,channels, ...
     'PyramidLevels',1);
figure
 I = imtile(I,'ThumbnailSize',[64 64]);
 imshow(I)
 title(['Layer ',name,' Features'],'Interpreter','none')
