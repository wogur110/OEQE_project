%% image load
roadimage=imread('imageset/roadimage.png');
roadimage=imresize(roadimage,0.2);
% roadimage=im2double(roadimage);
roaddepthmap=imread('imageset/roaddepthmap.png');
roaddepthmap=rgb2gray(roaddepthmap);
roaddepthmap=imresize(roaddepthmap,0.2);
% roaddepthmap=im2double(roaddepthmap);
res_road=[size(roadimage,1) size(roadimage,2)];

%% image slicing
% h=uint8(zeros(res_road(1), res_road(2), 3));
% image=uint8(zeros(res_road(1),res_road(2), 3));
% 
% for n=0:255 % n이 클수록 더 가까이 있는 것이다. n이 0이면 거리가 무한대에 가까운데. 일단 1을 더해 짱큰거리의 역수로 둔다. (질문한 다음 해결하자)
%     % d = 237/n*1e3*mm; % depthmap에 따라 거리를 정해야 하는데 일단 아무렇게나 정했다.
%         roaddepthmap_stack=repmat(uint8(roaddepthmap==n),1,1,3);
%         h=roadimage.*roaddepthmap_stack;
%         image = image + h;
% end

%%
% figure(1);
% subplot(1,2,1);
% imshow(roadimage);
% title('original image');
% subplot(1,2,2);
% imshow(image);
% title('should be same');
% %imwrite(image, ['test.png']);

%%
h=uint8(zeros(res_road(1), res_road(2), 3));
image=uint8(zeros(res_road(1),res_road(2), 3));
filter=[0 -1 0; -1 4 -1; 0 -1 0];
for n=0:255 % n이 클수록 더 가까이 있는 것이다. n이 0이면 거리가 무한대에 가까운데. 일단 1을 더해 짱큰거리의 역수로 둔다. (질문한 다음 해결하자)
    % d = 237/n*1e3*mm; % depthmap에 따라 거리를 정해야 하는데 일단 아무렇게나 정했다.
        roaddepthmap_stack=repmat(uint8(roaddepthmap==n),1,1,3);
        h=roadimage.*roaddepthmap_stack;
        for i=1:3
            image(:,:,i)=image(:,:,i)+uint8(conv2(h(:,:,i),filter,'same'));
        end
end

%% conv2 test
figure(1);
subplot(1,2,1);
imshow(roadimage);
title('original image');
subplot(1,2,2);
imshow(image);
title('convolution');
%imwrite(image, ['test.png']);
