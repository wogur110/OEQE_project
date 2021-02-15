%% basic size 
mm=10^-3; um=10^-6;
pupil_diameter=2*mm;
eye_focal_length = 21.64*mm; % 변수이다. 이게 바뀐다면 다 바뀐다. 16*mm(가까운 곳의 물체를 볼 때) ~ 24*mm(먼 곳의 물체를 볼 때)로 하자.
eye_length = 24*mm;
%% image load
roadimage=imread('imageset/roadimage.png');
roadimage=imresize(roadimage,0.2);
roadimage=im2double(roadimage);
roaddepthmap=imread('imageset/roaddepthmap.png');
roaddepthmap=rgb2gray(roaddepthmap);
roaddepthmap=imresize(roaddepthmap,0.2);
% roaddepthmap=im2double(roaddepthmap);
res_road=[size(roadimage,1) size(roadimage,2)];

%% point spread function 
b = @(distance) pupil_diameter * abs(eye_length*(1/eye_focal_length - 1/distance)-1); % 이 d는 depth map에 의해서 구해야 하는구나.
psf = @(c,r,d) 2/(pi() * (c * b(d))^2) * exp(-2*r^2/(c * b(d))^2);
%% separate by depth and convolution : depthmap이 각 pixel의 색깔을 
Res_window=[21 21];
wdx=0.001*mm; wdy=wdx;
wx = -floor(Res_window(2)/2)*wdx : wdx : floor(Res_window(2)/2)*wdx;
wy = -floor(Res_window(1)/2)*wdy : wdy : floor(Res_window(1)/2)*wdy;
window_radius = Res_window(2)/2*wdx;
[WX, WY] = meshgrid(wx,wy);
h=double(zeros(res_road(1), res_road(2), 3));
image=double(zeros(res_road(1),res_road(2), 3));
for n=0:255 % n이 클수록 더 가까이 있는 것이다. n이 0이면 거리가 무한대에 가까운데. 일단 1을 더해 짱큰거리의 역수로 둔다. (질문한 다음 해결하자)
    d = (256-n)*10*mm; % depthmap에 따라 거리를 정해야 하는데 일단 아무렇게나 정했다.
    for i=1:3
        h(:,:,i)=roadimage(:,:,i).*double(roaddepthmap==n);
        Window= WX.^2 + WY.^2 <= window_radius^2;
        Window= Window.*psf(1,sqrt(WX.^2+WY.^2), d);
        if sum(Window,'all')==0 % 이건 너무 기괴하다..n==234일 때 psf(1,0,d)는 1e+14를 내보내는데 어째서 Window에서의 값은 전부 0인가?! 어째서 인위적으로 적어줘야 하는가?
%             disp(n);
%             disp(d);
            Window(11,11)=1;
        else
            Window = Window / sum(Window, 'all');
        end
        image(:,:,i) = image(:,:,i) + double(conv2(h(:,:,i), Window, 'same'));
    end
end

normalize_factor = max(image,[],'all');
image = image / normalize_factor;
% %% image plotting
figure(1);
subplot(1,2,1);
imshow(roadimage);
title('original image');
subplot(1,2,2);
imshow(image);
title('blurred image');
%imwrite(image, ['test.png']);
% 
% % 지금 있는 문제: roaddepthmap==255일 때의 값이 튀어나오는데..? 그리고 애초에
% % depthmap을 이용해서 이렇게 자르면 되는 건가..????? 
% %% optical transfer function (:version 2: using fourier transform)
% % 어디서부터 시작하지..