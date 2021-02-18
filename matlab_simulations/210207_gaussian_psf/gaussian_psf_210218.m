%% basic size 
mm=10^-3; um=10^-6;
pupil_diameter=2*mm;
eye_focal_length = 21.64*mm; % 16mm when seeing near object, 24mm when seeing far object
eye_length = 24*mm; %fixed variable
%% image load
roadimage=imread('imageset/roadimage.png');
roadimage=imresize(roadimage,0.2);
roadimage=im2double(roadimage);
roaddepthmap=imread('imageset/roaddepthmap.png');
roaddepthmap=rgb2gray(roaddepthmap);
roaddepthmap=imresize(roaddepthmap,0.2);
res_road=[size(roadimage,1) size(roadimage,2)];

%% point spread function 
b = @(distance) pupil_diameter * abs(eye_length*(1/eye_focal_length - 1/distance)-1);
psf = @(c,r,d) 2/(pi() * (c * b(d))^2) * exp(-2*r^2/(c * b(d))^2);

%% separate by depth and convolution
Res_window=[21 21];
wdx=0.001*mm; wdy=wdx;
wx = -floor(Res_window(2) / 2) * wdx : wdx : floor(Res_window(2) / 2) * wdx;
wy = -floor(Res_window(1) / 2) * wdy : wdy : floor(Res_window(1) / 2) * wdy;
window_radius = floor(Res_window(2) / 2) * wdx;
[WX, WY] = meshgrid(wx,wy);
h = double(zeros(res_road(1), res_road(2), 3));
image = double(zeros(res_road(1),res_road(2), 3));
for n = 0:255 
    d = (256-n)*10*mm; %Convert depth image to real depth
    for i = 1:3
        h(:,:,i) = roadimage(:,:,i) .* double(roaddepthmap==n);
        Window = zeros(Res_window);
        if b(d) < 5e-06     %If b(d) is too small and cannot calculate psf -> Window is delta function
            %disp(n);
            Window(floor(Res_window(1) / 2) + 1, floor(Res_window(2) / 2) + 1) = 1; % delta function
        else
            Window = WX.^2 + WY.^2 <= window_radius^2;
            Window = Window.*psf(1,sqrt(WX.^2+WY.^2), d);
            Window = Window / sum(Window, 'all');   % psf function
        end
        image(:,:,i) = image(:,:,i) + double(conv2(h(:,:,i), Window, 'same'));  %Convolution
    end
end

normalize_factor = max(image,[],'all');
image = image / normalize_factor;

%% image plotting
figure(1);
subplot(1,2,1);
imshow(roadimage);
title('original image');
subplot(1,2,2);
imshow(image);
title('blurred image');
imwrite(image, ['result.png']);