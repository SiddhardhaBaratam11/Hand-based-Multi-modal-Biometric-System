clc; clear all; close all;

imds1 = imageDatastore('D:\education\notes\sem-4\mmp project\knuckle\exp\Database\vein',... %(After splitting)
     'IncludeSubfolders',true,...
     'LabelSource','foldernames');% Load the two datasets
knuckleData = load('feat_knuckle_squeezenet.mat');  % Load the knuckle features dataset
veinData = load('feat_vein_squeezenet.mat');  % Load the vein features dataset

% Extract the data from the loaded structures
% You may want to inspect the structure of knuckleData and veinData to find the actual field names
knuckleFields = fieldnames(knuckleData);
veinFields = fieldnames(veinData);

% Assuming that both structures have only one field, you can proceed as follows
if numel(knuckleFields) == 1 && numel(veinFields) == 1
    feat_knuckle = knuckleData.(knuckleFields{1});
    feat_vein = veinData.(veinFields{1});

    % Check the sizes of the datasets
    [rows_knuckle, cols_knuckle] = size(feat_knuckle);
    [rows_vein, cols_vein] = size(feat_vein);

    % Ensure that the datasets have the same number of rows
    if rows_knuckle > rows_vein
        % Trim feat_knuckle to have the same number of rows as feat_vein
        feat_knuckle = feat_knuckle(1:rows_vein, :);
    elseif rows_knuckle < rows_vein
        % Trim feat_vein to have the same number of rows as feat_knuckle
        feat_vein = feat_vein(1:rows_knuckle, :);
    end

    % Combine the datasets by horizontally concatenating them
    combined_features = [feat_knuckle, feat_vein];

    % Select the first 500 rows and the first 19600 columns
    combined_features = combined_features(1:500, 1:19600);

    % Save the modified combined dataset to a new .mat file
    save('combined_features1_500x19600.mat', 'combined_features');
else
    error('The structure of the .mat files is not as expected.');
end


%Encryption
% Load the feature data from the file
features = load('combined_features1_500x19600.mat');


% Load the encryption key from a PEM file
encryptionKey = 'MIIBIjANBg';
%encryptionKey = fileread(encryptionKeyFile);

% Save the feature data to a file
save('features_data2.mat', 'features');

% Encryption
system(['openssl enc -aes-256-cbc -a -salt -pbkdf2 -k "', encryptionKey, '" -in "features_data2.mat" -out "encrypted_features_data2.enc"']);
disp('AES encryption successful.');

% Decryption
[status, ~] = system(['openssl enc -d -aes-256-cbc -a -pbkdf2 -k "', encryptionKey, '" -in "encrypted_features_data2.enc" -out "decrypted_features_data2.mat"']);

if status == 0
    disp('AES decryption successful.');
else
    error('AES decryption failed.');
end

% Load the decrypted feature data from the file
decryptedFeatureData = load('decrypted_features_data2.mat');
decryptedFeatures = decryptedFeatureData.features;
 
%classification
load('combined_features_500x19600.mat')
labels1 = imds1.Labels;
% Measure network accuracy for XRAY DB 
Model=fitcecoc(combined_features(2:2:500,:),labels1(2:2:500,:));
predict_1=predict(Model,combined_features(1:2:500,:));
acc=mean(labels1(1:2:500,:)==predict_1);
confusionchart(labels1(1:2:500,:),predict_1)
plotconfusion(labels1(1:2:500,:),predict_1)