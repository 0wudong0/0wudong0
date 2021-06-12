clc;
clear all;
close all;
addpath(genpath('image'));
addpath(genpath('whyte_code'));
addpath(genpath('cho_code'));
opts.prescale = 1; %%downsampling
opts.xk_iter = 5; %% the iterations
opts.gamma_correct = 1.0;
opts.k_thresh = 20;

Start = 1;
End = 500;

volume_size = End - Start+1;

filename = strcat('Z:\Master theses\Toon - PoD\from SIMON\cooler_2000_1\',sprintf('slice_%04d',Start),'.tif');
[ image_u  image_v] = size(imread(filename));

volume_ori = zeros(image_u, image_v, volume_size );
volume_new = zeros(image_u, image_v, volume_size );

% max_min_ori  = zeros(2,volume_size);
% max_min_new  = zeros(2,volume_size);



for number = Start:End

% parameters
number

% filename = strcat('Z:\Master theses\Toon - PoD\from SIMON\manifold_2000_1\',sprintf('slice_%04d',number),'.tif'); %243
% filename = strcat('Z:\Master theses\Toon - PoD\from SIMON\turbine_2000_1\',sprintf('slice_%04d',number),'.tif'); % 700
filename = strcat('Z:\Master theses\Toon - PoD\from SIMON\cooler_2000_1\',sprintf('slice_%04d',number),'.tif'); % 583

% opts.kernel_size = 75;  saturation = 0;
% % filename = strcat('Z:\Master  theses\Toon - PoD\from SIMON\turbine_2000_1\',sprintf('slice_%04d',number),'.tif'); opts.kernel_size = 13;  saturation = 0;
% lambda_pixel = 4e-3; lambda_grad = 4e-3; opts.gamma_correct = 2.2;
% lambda_tv = 0.002; lambda_l0 = 2e-4; weight_ring = 1;

opts.kernel_size = 15;  saturation = 0;
lambda_pixel = 4e-3; lambda_grad = 4e-3; opts.gamma_correct = 2.2;
lambda_tv = 0.002; lambda_l0 = 2e-4; weight_ring = 0;

% opts.kernel_size = 95;  saturation = 1;
% lambda_pixel = 4e-3; lambda_grad = 4e-3; opts.gamma_correct = 2.2;

y = imread(filename);
max_y = max(y(:));
min_y = min(y(:));


% local image with manual selection
top = 1;
bottom = floor(size(y,1));
% bottom = 450;
left = 1;
right = size(y,2);
y = y(top:bottom,left:right);

isselect = 0; %false or true, if a local image is needed
if isselect ==1
    figure, imshow(y);
    %tips = msgbox('Please choose the area for deblurring:');
    fprintf('Please choose the area for deblurring:\n');
    h = imrect;
    position = wait(h);
    close;
    B_patch = imcrop(y,position);
    y = (B_patch);
else
    y = y;
end
if size(y,3)==3
    yg = im2double(rgb2gray(y));
else
    yg = im2double(y);
end


volume_ori(:,:,number-Start+1) = yg;
% max_min_ori(1,number-Start+1) = max(yg(:));
% max_min_ori(2,number-Start+1) = min(yg(:));
% blind deconvolution
tic;
[kernel, interim_latent] = blind_deconv(yg, lambda_pixel, lambda_grad, opts);
toc
y = im2double(y);
%% Final Deblur: 
if ~saturation
    %% 1. TV-L2 denoising method
    Latent = ringing_artifacts_removal(y, kernel, lambda_tv, lambda_l0, weight_ring);
else
    %% 2. Whyte's deconvolution method (For saturated images)
    Latent = whyte_deconv(y, kernel);
end

% figure,subplot(1,2,1);imshow(y,[]);subplot(1,2,2);imshow(Latent,[]);
% figure,plot(y(:,100));hold on ; plot(Latent(:,100));legend('ori','new')

% figure,plot(y(200,:));hold on ; plot(Latent(200,:));legend('ori','new')
%   fprintf('the minimum of Latent is %f\n',min(Latent(:)));
%% save kernel
k = kernel - min(kernel(:));
k = k./max(k(:));
imwrite(k,strcat('results\',  sprintf('slice_%04d',number),'.png'));
% imwrite(k,strcat('results\', num2str(number),'_',num2str(opts.kernel_size ), '_kernel.png'));

%% save image
max_Latent=max(Latent(:));
min_Latent=min(Latent(:));
% Latent_out =Latent./max_Latent*65535;
% max_min_new(1,number-Start+1) = max_Latent;
% max_min_new(2,number-Start+1) = min_Laten;

% imwrite(uint16(Latent_out),strcat('results\',  sprintf('slice_%04d',number),'-',num2str(opts.kernel_size ),'-',num2str(lambda_tv),'-',num2str(lambda_l0),'-',num2str(weight_ring),'.tif'));


%%   fprintf('the minimum of S is %f\n',min(S(:)));

%% save volume data
volume_new(:,:,number-Start+1) = Latent;
end
% 
save volume_ori.mat  volume_ori
save volume_new.mat  volume_new