%% retina to display method : to eliminate square-like black regions
%%
um =10^-6; mm= 10^-3; cm = 10^-2;
eye_length = 20*mm;
FOVx = 30*pi/180;
FOVy = 30*pi/180;
light_field_origin_plane_distance = 1;
source = './imageset';
result = './result';
depth = [0.2, 0.3, 1.0, 2.0, 5.0];
%%
res_world = [200 200];
wdx = 2*light_field_origin_plane_distance*tan(FOVx/2)/res_world(2);
wdy = 2*light_field_origin_plane_distance*tan(FOVy/2)/res_world(1);
wx = -res_world(2)/2*wdx + wdx/2 : wdx : res_world(2)/2*wdx - wdx/2;
wy = -res_world(1)/2*wdy + wdy/2 : wdy : res_world(1)/2*wdy - wdy/2;

res_view = [7 7];
view_radius = 1.5*mm;
vdx = 0.5*mm; vdy = vdx;
vx = -res_view(2)/2*vdx + vdx/2 : vdx : res_view(2)/2*vdx - vdx/2;
vy = -res_view(1)/2*vdy + vdy/2 : vdy : res_view(1)/2*vdy - vdy/2;
[VY, VX] = meshgrid(vy,vx);
Pupil = VX.^2+VY.^2<=view_radius^2;

res_retina = [200 200];
rdx = 0.05*mm; rdy = rdx;
rx = -res_retina(2)/2*rdx+rdx/2 : rdx : res_retina(2)/2*rdx - rdx/2;
ry = -res_retina(1)/2*rdy+rdy/2 : rdy : res_retina(1)/2*rdy - rdy/2;

%% get world images
first = 1;
last = 49;
images = zeros(last, res_world(1), res_world(2), 3);
for imgid = first : last
     fileLocator = fullfile(source, sprintf('%04d.png', imgid)) ;
     if ~exist(fileLocator, 'file'),  continue ;  end
     img = imread(fileLocator, 'PNG');
     img = imresize(img, res_world);
     images(imgid,:,:,:) = img;
end

%% Ray initialization
res_ray = prod(res_view .* res_world); % total number of rays
rays = zeros(res_ray,7); %% (Ray information : x,theta_x,y,theta_y / retina_x, retina_y, view_idx / OutofIndex)
for vy_idx = 1:res_view(1)
    for vx_idx = 1:res_view(2)
        view_idx = (vy_idx-1)*res_view(2)+vx_idx;
        idx = (view_idx-1) * prod(res_world) + 1:prod(res_retina);
        ry_idx = fix((1:prod(res_retina))/res_retina(2));
        rx_idx = rem(1:prod(res_retina),res_retina(1));
        for ry_idx = 1:res_retina(1)
            for rx_idx = 1:res_retina(2)
                retina_idx = (ry_idx - 1) * res_retina(2) + rx_idx;
                idx = (view_idx-1) * prod(res_world) + retina_idx;
                rays(idx,1)=rx(rx_idx);
                rays(idx,2)=atan((vx(vx_idx)-rx(rx_idx))/eye_length);
                rays(idx,3)=ry(ry_idx);
                rays(idx,4)=atan((vy(vy_idx)-ry(ry_idx))/eye_length);
                rays(idx,5)=rx_idx;
                rays(idx,6)=ry_idx;
                rays(idx,7)=vx_idx;
                rays(idx,8)=vy_idx;
                rays(idx,9)=Pupil(vy_idx, vx_idx);
            end
        end
    end
end

for i = 1:length(depth)
    focal_length = 1/(1/eye_length + 1/depth(i));
    %% Ray transfer matrix
    transfer_x = [1 light_field_origin_plane_distance; 0 1] * [1 0; -1/focal_length 1] * [1 eye_length; 0 1];
    transfer_y = transfer_x;

    transfer = blkdiag(transfer_x, transfer_y, eye(5));
    %%
    ray_on_retina = (transfer * rays')';
    retina_image = zeros(res_retina(1), res_retina(2), 3);
    
    for idx=1:res_ray
        ray = ray_on_retina(idx,:);
        wx = ray(1);
        wy = ray(3);
        rx_idx = ray(5);
        ry_idx = ray(6);
        vx_idx = ray(7);
        vy_idx = ray(8);
        view_idx = (vy_idx-1)*res_view(2) + vx_idx;
        wx_idx = floor((wx-vx(vx_idx))/wdx+0.5+res_world(2)/2);
        wy_idx = floor((wy-vy(vy_idx))/wdy+0.5+res_world(1)/2);
        if wx_idx >= 1 && wx_idx <= res_world(2) && wy_idx >= 1 && wy_idx <= res_world(1)
            retina_image(ray(6),ray(5),:) = retina_image(ray(6),ray(5),:)+ray(9)*reshape(images(view_idx,wy_idx,wx_idx,:),[1,1,3]);
        end
    end
    %% show image
    retina_image = uint8(retina_image / nnz(Pupil));
    retina_image = rot90(retina_image,2);
    figure(i);
    imshow(retina_image);
    title(sprintf('retina%.2f',depth(i)));
    imwrite(retina_image, fullfile(result, sprintf('result%.2f.png',depth(i))));
end




