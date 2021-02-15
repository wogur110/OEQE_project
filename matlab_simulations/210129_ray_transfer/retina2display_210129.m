%% 210129_geo_optics_sim
um =10^-6; mm= 10^-3; cm= 10^-2;

%% Image Load
target = im2double(imread('TOY1.png'));

%% Retina
Res_retina = [size(target,1) size(target,2)];
rdx = 0.05*mm; rdy = rdx;
rx = -Res_retina(2)/2*rdx + rdx/2 : rdx : Res_retina(2)/2*rdx - rdx/2;
ry = -Res_retina(1)/2*rdy + rdy/2 : rdy : Res_retina(1)/2*rdy - rdy/2;
[RX RY] = meshgrid(rx,ry);
length_eye = 16*mm;

%% Pupil
Res_pupil = [5 5];
pdx = 1*mm; pdy = pdx;
pupil_radius = 2*mm;
px = -Res_pupil(2)/2*pdx + pdx/2 : pdx : Res_pupil(2)/2*pdx - pdx/2;
py = -Res_pupil(1)/2*pdy + pdy/2 : pdy : Res_pupil(1)/2*pdy - pdy/2;
[PX PY] = meshgrid(px,py);
Pupil = PX.^2+PY.^2<=pupil_radius^2;
%Pupil = Pupil .* fspecial('gaussian', Res_pupil, 0.5);
length_focal = 20*cm;

%% Display
Res_display = [size(target,1) size(target,2)];
pp_display = 1*mm;
dx = -Res_display(2)/2*pp_display + pp_display/2 : pp_display : Res_display(2)/2*pp_display - pp_display/2;
dy = -Res_display(1)/2*pp_display + pp_display/2 : pp_display : Res_display(1)/2*pp_display - pp_display/2;
length_display = 25*cm;

%% Ray initialization
Res_ray = prod(Res_pupil .* Res_retina); %% Retina --> Display : prod(Res_pupil .* Res_retina) / Display --> Retina : prod(Res_pupil .* Res_display)
Ray = zeros(Res_ray,8); %% (Ray information : x,y,theta_x,theta_y / Display information : R, G, B / OutOfIndex)
for i=1:Res_retina(1)
    for j =1:Res_retina(2)
        retina_idx = Res_retina(2)*(i-1) + j;
        idx = (retina_idx-1)*prod(Res_pupil)+1 : retina_idx*prod(Res_pupil);
        Ray(idx,1) = rx(j);
        Ray(idx,2) = atan((PX(:)-rx(j))/length_eye);
        Ray(idx,3) = ry(i);        
        Ray(idx,4) = atan((PY(:)-ry(i))/length_eye);
        Ray(idx,5) = target(i,j,1);
        Ray(idx,6) = target(i,j,2);
        Ray(idx,7) = target(i,j,3);
        Ray(idx,8) = Pupil(:);
    end
end

%% Ray transfer matrix
TM_x = [1 length_display; 0 1] * [1 0; -1/length_focal 1] * [1 length_eye; 0 1];
TM_y = [1 length_display; 0 1] * [1 0; -1/length_focal 1] * [1 length_eye; 0 1];

%% Ray .* Ray transfer matrix
TM_total = blkdiag(TM_x, TM_y, eye(4));

%% Generate sparse matrix (Matrix size : Res_ray*Res_display X Res_display)
Ray_after = (TM_total * Ray')';
DIsplay_Image = zeros(Res_display(1), Res_display(2), 3);
for idx = 1 : Res_ray 
    Ray_idx = Ray_after(idx,:);
    x_idx = Ray_idx(1);
    y_idx = Ray_idx(3);
    %x_coord = floor( (-Res_display(2)/2*pp_display + pp_display/2)) / pp_display) + 1;
    %y_coord = floor((y_idx - (-Res_display(1)/2*pp_display + pp_display/2)) / pp_display) + 1;
    x_coord = (x_idx -(-Res_display(2)/2*pp_display + pp_display/2)) / pp_display;
    x_coord_floor = floor(x_coord); interpolate_rate_x = x_coord - x_coord_floor;
    y_coord = (y_idx - (-Res_display(1)/2*pp_display + pp_display/2)) / pp_display;
    y_coord_floor = floor(y_coord); interpolate_rate_y = y_coord - y_coord_floor;
    if x_coord > 0 && x_coord <= Res_display(2) && y_coord > 0 && y_coord <= Res_display(1)
        %DIsplay_Image(y_coord, x_coord, :) = DIsplay_Image(y_coord, x_coord, :) + reshape(Ray_idx(5:7) * Ray_idx(8), [1,1,3]);
        DIsplay_Image(y_coord_floor,x_coord_floor,:) = DIsplay_Image(y_coord_floor,x_coord_floor,:) + reshape(Ray_idx(5:7) * Ray_idx(8) * (1-interpolate_rate_x) * (1-interpolate_rate_y), [1,1,3]);
        DIsplay_Image(y_coord_floor + 1,x_coord_floor,:) = DIsplay_Image(y_coord_floor + 1,x_coord_floor,:) + reshape(Ray_idx(5:7) * Ray_idx(8) * (1-interpolate_rate_x) * (interpolate_rate_y), [1,1,3]);
        DIsplay_Image(y_coord_floor,x_coord_floor + 1,:) = DIsplay_Image(y_coord_floor,x_coord_floor + 1,:) + reshape(Ray_idx(5:7) * Ray_idx(8) * (interpolate_rate_x) * (1-interpolate_rate_y), [1,1,3]);
        DIsplay_Image(y_coord_floor + 1,x_coord_floor + 1,:) = DIsplay_Image(y_coord_floor + 1,x_coord_floor + 1,:) + reshape(Ray_idx(5:7) * Ray_idx(8) * (interpolate_rate_x) * (interpolate_rate_y), [1,1,3]);
    end
end

%% Plot Image of Display
normalize_factor = max(DIsplay_Image,[],'all');
DIsplay_Image = DIsplay_Image / normalize_factor;

figure(1);
title_txt = strcat('length eye : ', string(length_eye), ', length dp : ', string(length_display), ', length focal : ', string(length_focal));
sgtitle(['Retina -> Display', title_txt]);
subplot(1,2,1);
imshow(target);
title('Retina');
subplot(1,2,2);
imshow(DIsplay_Image);
title('Display');
    
resize_ratio = (rdx / length_eye) / (pp_display / length_display) 
origin_shape = [size(target,1) size(target,2)]
resize_shape = origin_shape * resize_ratio
rect = [origin_shape(2) / 2 - resize_shape(2) / 2, origin_shape(1) / 2 - resize_shape(2) / 2, resize_shape(2), resize_shape(1)]
Display_Image_crop  = imcrop(DIsplay_Image(:,:,1:3), rect);
Display_Image_reshape = imrotate(imresize(Display_Image_crop, origin_shape),180);
peaksnr = psnr(Display_Image_reshape, target)
ssimval = ssim(Display_Image_reshape, target)
