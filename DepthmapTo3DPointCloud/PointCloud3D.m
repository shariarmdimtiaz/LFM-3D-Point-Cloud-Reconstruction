close all; 
clc; clear;

% Read/load depth map
% N.B. Change the directory as needed
depthImg = imread('depthmap-hci\boxes_depth.png');
grayI = rgb2gray(depthImg);
% figure; imshow(grayI);
depthmap = grayI;

% Flip
% depthmap = flip(depthmap, 1);
% depthmap = flip(depthmap, 2);

s=size(depthmap);


% Color map
cm = (depthmap * (100/255)); %  255 ¿¡¼­ 100 


Z = double(cm);
[X, Y] = meshgrid(1:s(1), 1:s(2));
obj=depthimg2point(Z, 0);
obj(:, 3)=obj(:, 3); 

ptCloud = pointCloud(obj);
figure; pcshow(ptCloud);
xlabel('X');
ylabel('Y');
zlabel('Z');
% title('3D Point Cloud')
axis ('equal')
ax = gca;
ax.XColor = 'black' ; % Red
ax.YColor = 'black' ; % Blue
ax.ZColor = 'black';
set(gca, 'color', 'w')
set(gcf,'color','w');

% Save PLY format
% pcwrite(ptCloud,'PointCloud3D','PLYFormat','binary');

