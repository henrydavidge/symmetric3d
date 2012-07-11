function [f_c,f_r,f_b] = register_images_3D_symmetric(img1,img2)

%Create two test images
imgSize= 36;
% [x,y,z] = meshgrid(-18:17,-18:17,-18:17);
% r = x.^2 + y.^2 + z.^2;
% img1=zeros(imgSize,imgSize,imgSize);
% img1(r <= 15^2)=1;
% img1=filter_image(img1,5);
% img2=zeros(imgSize,imgSize,imgSize);
% img2(r <= 10^2)=1;
% img2=filter_image(img2,5);
[x,y,z] = meshgrid(1:imgSize,1:imgSize,1:imgSize);
%Display the images
% figure('Visible', 'off');
% subplot(1,2,1);
% %imshow(mat2gray(img1(:,:,18)));
% slice(x,y,z,img1,[],[],[0,8,18,28,36]);
% subplot(1,2,2);
% %imshow(mat2gray(img2(:,:,18)));
% slice(x,y,z,img2,[],[],[0,8,18,28,36]);
% print 'out/orig.pdf' -dpdf

%Write Spider files
writeSPIDERfile('out/img1.spi', img1);
writeSPIDERfile('out/img2.spi', img2);

disp('Original Error:');
disp(sum((img1(:) - img2(:)).^2));


%Register the images
%Initialize the registration function
[fCol,fRow,fBea]=meshgrid(1:imgSize,1:imgSize,1:imgSize);
[gCol,gRow,gBea] = meshgrid(1:imgSize,1:imgSize,1:imgSize);
%Call the registration function 
tic;
[fCol1,fRow1,fBea1,gCol1,gRow1,gBea1]=register_images(img1,img2,fCol,fRow,fBea,gCol,gRow,gBea,.2,1000);
% [fCol2,fRow2,fBea2,gCol2,gRow2,gBea2]=register_images(img2,img1,fCol,fRow,fBea,gCol,gRow,gBea,.2,1000);
% fCol1 = gather(fCol1);
% fRow1 = gather(fRow1);
% fBea1 = gather(fBea1);
% gCol2 = gather(gCol2);
% gRow2 = gather(gRow2);
% gBea2 = gather(gBea2);
save('out/out.mat');
toc;
   
   


function [f_c,f_r,f_b,g_c,g_r,g_b]=register_images(img1,img2,f_c,f_r,f_b,g_c,g_r,g_b,rho,maxIter)
lambda = .01;
lambda2 = -.01;
f_c_init=f_c;
f_r_init=f_r;
f_b_init = f_b;

g_c_init=g_c;
g_r_init=g_r;
g_b_init = g_b;
imgSize=36;
stopevery = 200;

[dr_img1_orig,dc_img1_orig,db_img1_orig]=grad_img(img1);
[dr_img2_orig,dc_img2_orig,db_img2_orig]=grad_img(img2);

%compile cuda file%
if (exist('symmetric3d.ptx','file') ~= 2)
   if (ispc)
       [status, result] = system('"C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\vcvarsall.bat" & nvcc -ptx symmetric3d.cu','-echo');
   else
       [status, result] = system('nvcc -ptx symmetric3d.cu','-echo');
   end
end

%initialize kernels%
interp = parallel.gpu.CUDAKernel('symmetric3d.ptx', 'symmetric3d.cu', 'interp3');
interp.ThreadBlockSize = [6,6,6];
interp.GridSize = [6,36];

extf = parallel.gpu.CUDAKernel('symmetric3d.ptx', 'symmetric3d.cu', 'extf');
extf.ThreadBlockSize = [6,6,6];
extf.GridSize = [6,36];

jacPartialsAndBarrier = parallel.gpu.CUDAKernel('symmetric3d.ptx', 'symmetric3d.cu', 'jacPartials');
jacPartialsAndBarrier.ThreadBlockSize = [6,6,6];
jacPartialsAndBarrier.GridSize = [6,36];

intf = parallel.gpu.CUDAKernel('symmetric3d.ptx', 'symmetric3d.cu', 'intf');
intf.ThreadBlockSize = [6,6,6];
intf.GridSize = [6,36];

jacobian = parallel.gpu.CUDAKernel('symmetric3d.ptx', 'symmetric3d.cu', 'jacobian');
jacobian.ThreadBlockSize = [6,6,6];
jacobian.GridSize = [6,36];

d_f = parallel.gpu.CUDAKernel('symmetric3d.ptx', 'symmetric3d.cu', 'd_f');
d_f.ThreadBlockSize = [6,6,6];
d_f.GridSize = [6,36];

%initialize variables
img2_o_f = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
jacf = parallel.gpu.GPUArray.ones(imgSize, imgSize, imgSize);

img1_o_g = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
jacg = parallel.gpu.GPUArray.ones(imgSize, imgSize, imgSize);

f_c = gpuArray(f_c);
dc_img2 = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);

d_f_c = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);

i_m_1xf = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
i_p_1xf = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
j_m_1xf = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
j_p_1xf = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
k_m_1xf = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
k_p_1xf = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);

barrierxf = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);

extf_c = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
intf_c = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);

f_r = gpuArray(f_r);
dr_img2 = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);

d_f_r = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);

i_m_1yf = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
i_p_1yf = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
j_m_1yf = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
j_p_1yf = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
k_m_1yf = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
k_p_1yf = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);

barrieryf = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);

extf_r = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
intf_r = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);

f_b = gpuArray(f_b);
db_img2 = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);

d_f_b = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);

i_m_1zf = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
i_p_1zf = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
j_m_1zf = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
j_p_1zf = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
k_m_1zf = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
k_p_1zf = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);

barrierzf = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);

extf_b = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
intf_b = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);

g_c = gpuArray(g_c);
dc_img1 = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);

d_g_c = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);

i_m_1xg = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
i_p_1xg = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
j_m_1xg = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
j_p_1xg = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
k_m_1xg = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
k_p_1xg = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);

barrierxg = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);

extg_c = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
intg_c = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);

g_r = gpuArray(g_r);
dr_img1 = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);

d_g_r = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);

i_m_1yg = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
i_p_1yg = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
j_m_1yg = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
j_p_1yg = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
k_m_1yg = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
k_p_1yg = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);

barrieryg = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);

extg_r = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
intg_r = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);

g_b = gpuArray(g_b);
db_img1 = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);

d_g_b = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);

i_m_1zg = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
i_p_1zg = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
j_m_1zg = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
j_p_1zg = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
k_m_1zg = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
k_p_1zg = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);

barrierzg = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);

extg_b = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);
intg_b = parallel.gpu.GPUArray.zeros(imgSize, imgSize, imgSize);

for k=1:maxIter,
    %Interpolate image1
    img1_o_g = feval(interp, img1_o_g, g_c, g_r, g_b, img1, imgSize, imgSize, imgSize);
    dr_img1 = feval(interp, dr_img1, g_c, g_r, g_b, dr_img1_orig, imgSize, imgSize, imgSize);
    dc_img1 = feval(interp, dc_img1, g_c, g_r, g_b, dc_img1_orig, imgSize, imgSize, imgSize);
    db_img1 = feval(interp, db_img1, g_c, g_r, g_b, db_img1_orig, imgSize, imgSize, imgSize);
    
    %Interpolate image2
    img2_o_f = feval(interp, img2_o_f, f_c, f_r, f_b, img2, imgSize, imgSize, imgSize);
    dr_img2 = feval(interp, dr_img2, f_c, f_r, f_b, dr_img2_orig, imgSize, imgSize, imgSize);
    dc_img2 = feval(interp, dc_img2, f_c, f_r, f_b, dc_img2_orig, imgSize, imgSize, imgSize);
    db_img2 = feval(interp, db_img2, f_c, f_r, f_b, db_img2_orig, imgSize, imgSize, imgSize);

    %external energies
    extg_r = -feval(extf, extg_r, img1_o_g, img2_o_f, dr_img1, imgSize, imgSize, imgSize); 
    extg_c = -feval(extf, extg_c, img1_o_g, img2_o_f, dc_img1, imgSize, imgSize, imgSize);
    extg_b = -feval(extf, extg_b, img1_o_g, img2_o_f, db_img1, imgSize, imgSize, imgSize);
    
    extf_r = feval(extf, extf_r, img1_o_g, img2_o_f, dr_img2, imgSize, imgSize, imgSize); 
    extf_c = feval(extf, extf_c, img1_o_g, img2_o_f, dc_img2, imgSize, imgSize, imgSize);
    extf_b = feval(extf, extf_b, img1_o_g, img2_o_f, db_img2, imgSize, imgSize, imgSize);
    
    
    %again separating into three terms to avoid long lines
    [i_m_1xg, i_p_1xg, j_m_1xg, j_p_1xg, k_m_1xg, k_p_1xg, barrierxg] = feval(jacPartialsAndBarrier,...
        i_m_1xg, i_p_1xg, j_m_1xg, j_p_1xg, k_m_1xg, k_p_1xg, barrierxg,...
        jacg, g_r, g_b, img1_o_g, img2_o_f, imgSize, imgSize, imgSize, 1);
    
    [i_m_1yg, i_p_1yg, j_m_1yg, j_p_1yg, k_m_1yg, k_p_1yg, barrieryg] = feval(jacPartialsAndBarrier,...
        i_m_1yg, i_p_1yg, j_m_1yg, j_p_1yg, k_m_1yg, k_p_1yg, barrieryg,...
        jacg, g_c, g_b, img1_o_g, img2_o_f, imgSize, imgSize, imgSize, -1);
    
    [i_m_1zg, i_p_1zg, j_m_1zg, j_p_1zg, k_m_1zg, k_p_1zg, barrierzg] = feval(jacPartialsAndBarrier,...
        i_m_1zg, i_p_1zg, j_m_1zg, j_p_1zg, k_m_1zg, k_p_1zg, barrierzg,...
        jacg, g_c, g_r, img1_o_g, img2_o_f, imgSize, imgSize, imgSize, 1);
    
    [i_m_1xf, i_p_1xf, j_m_1xf, j_p_1xf, k_m_1xf, k_p_1xf, barrierxf] = feval(jacPartialsAndBarrier,...
        i_m_1xf, i_p_1xf, j_m_1xf, j_p_1xf, k_m_1xf, k_p_1xf, barrierxf,...
        jacf, f_r, f_b, img1_o_g, img2_o_f, imgSize, imgSize, imgSize, 1);
    
    [i_m_1yf, i_p_1yf, j_m_1yf, j_p_1yf, k_m_1yf, k_p_1yf, barrieryf] = feval(jacPartialsAndBarrier,...
        i_m_1yf, i_p_1yf, j_m_1yf, j_p_1yf, k_m_1yf, k_p_1yf, barrieryf,...
        jacf, f_c, f_b, img1_o_g, img2_o_f, imgSize, imgSize, imgSize, -1);
    
    [i_m_1zf, i_p_1zf, j_m_1zf, j_p_1zf, k_m_1zf, k_p_1zf, barrierzf] = feval(jacPartialsAndBarrier,...
        i_m_1zf, i_p_1zf, j_m_1zf, j_p_1zf, k_m_1zf, k_p_1zf, barrierzf,...
        jacf, f_c, f_r, img1_o_g, img2_o_f, imgSize, imgSize, imgSize, 1);
    
    
    
    
                    
    %Internal energies
    intg_c = feval(intf, intg_c, g_c, imgSize, imgSize, imgSize);
        
    intg_r = feval(intf, intg_r, g_r, imgSize, imgSize, imgSize);
        
    intg_b = feval(intf, intg_b, g_b, imgSize, imgSize, imgSize);
    
    intf_c = feval(intf, intf_c, f_c, imgSize, imgSize, imgSize);
        
    intf_r = feval(intf, intf_r, f_r, imgSize, imgSize, imgSize);
        
    intf_b = feval(intf, intf_b, f_b, imgSize, imgSize, imgSize);
    
    while(1)
        d_g_c = feval(d_f,d_g_c,jacf,jacg,extg_c,i_m_1xg,i_p_1xg,j_m_1xg,j_p_1xg,k_m_1xg,k_p_1xg,intg_c,barrierxg,rho,lambda,lambda2,imgSize,imgSize,imgSize);
%         d_f_c=min(max(d_f_c,-3),3); %don't want to jump too much in one iteration
        g_c_temp=g_c+d_g_c;
        
        g_c_temp = max(min(g_c_temp, imgSize), 1);
        
        
        
        
        d_g_r = feval(d_f,d_g_r,jacf,jacg,extg_r,i_m_1yg,i_p_1yg,j_m_1yg,j_p_1yg,k_m_1yg,k_p_1yg,intg_r,barrieryg,rho,lambda,lambda2,imgSize,imgSize,imgSize);
%         d_f_r=min(max(d_f_r,-3),3);
        g_r_temp=g_r+d_g_r;
        
        g_r_temp = max(min(g_r_temp, imgSize), 1);
        
        d_g_b = feval(d_f,d_g_b,jacf,jacg,extg_b,i_m_1zg,i_p_1zg,j_m_1zg,j_p_1zg,k_m_1zg,k_p_1zg,intg_b,barrierzg,rho,lambda,lambda2,imgSize,imgSize,imgSize);
%         d_f_b=min(max(d_f_b,-3),3);
        g_b_temp=g_b+d_g_b;
        
        g_b_temp = max(min(g_b_temp, imgSize), 1);
        
        
        d_f_c = feval(d_f,d_f_c,jacf,jacg,extf_c,i_m_1xf,i_p_1xf,j_m_1xf,j_p_1xf,k_m_1xf,k_p_1xf,intf_c,barrierxf,rho,lambda,lambda2,imgSize,imgSize,imgSize);
%         d_f_c=min(max(d_f_c,-3),3); %don't want to jump too much in one iteration
        f_c_temp=f_c+d_f_c;
        
        f_c_temp = max(min(f_c_temp, imgSize), 1);
        
        
        
        
        d_f_r = feval(d_f,d_f_r,jacf,jacg,extf_r,i_m_1yf,i_p_1yf,j_m_1yf,j_p_1yf,k_m_1yf,k_p_1yf,intf_r,barrieryf,rho,lambda,lambda2,imgSize,imgSize,imgSize);
%         d_f_r=min(max(d_f_r,-3),3);
        f_r_temp=f_r+d_f_r;
        
        f_r_temp = max(min(f_r_temp, imgSize), 1);
        
        d_f_b = feval(d_f,d_f_b,jacf,jacg,extf_b,i_m_1zf,i_p_1zf,j_m_1zf,j_p_1zf,k_m_1zf,k_p_1zf,intf_b,barrierzf,rho,lambda,lambda2,imgSize,imgSize,imgSize);
%         d_f_b=min(max(d_f_b,-3),3);
        f_b_temp=f_b+d_f_b;
        
        f_b_temp = max(min(f_b_temp, imgSize), 1);
        
        jacf_temp = feval(jacobian, jacf, f_c_temp, f_r_temp, f_b_temp, imgSize, imgSize, imgSize);
        jacg_temp = feval(jacobian, jacg, g_c_temp, g_r_temp, g_b_temp, imgSize, imgSize, imgSize);

        if (min(jacf_temp(:)) >= 0 && min(jacg_temp(:)) >= 0)
            g_c = g_c_temp;
            g_r = g_r_temp;
            g_b = g_b_temp;
            jacg = jacg_temp;
            
            f_c = f_c_temp;
            f_r = f_r_temp;
            f_b = f_b_temp;
            jacf = jacf_temp;
            break;
        else
            rho = .5 * rho;
%             disp(min(jac(:)));
        end
        
    end
    
    
    
end

% figure('Visible', 'off');
% subplot(1,2,1);
% slice(f_c_init,f_r_init,f_b_init,img2_o_f,[],[],[0,8,18,28,36]);
% %imshow(mat2gray(img2_o_f(:,:,12)));
% subplot(1,2,2);
% slice(g_c_init, g_r_init,g_b_init,img1_o_g,[],[],[0,8,18,28,36]);
% %imshow(mat2gray(tmp(:,:,12)));
% print 'out/slices.pdf' -dpdf
% 
% figure('Visible', 'off');
% tmp=-img1+img2_o_f;
% tmp(1,1)=1;
% slice(f_c_init,f_r_init,f_b_init,tmp,[],[],[0,8,18,28,36]);
% print 'out/diff.pdf' -dpdf
% 
% 
% figure('Visible', 'off');
% plot_profile(img1_o_g,img2_o_f,18);
% print 'out/profile.pdf' -dpdf

%Write final Spider files
img1_o_g = gather(img1_o_g);
img2_o_f = gather(img2_o_f);
writeSPIDERfile('out/img1_o_g.spi',img1_o_g);
writeSPIDERfile('out/img2_o_f.spi',img2_o_f);
writeSPIDERfile('out/diff.spi', img1_o_g - img2_o_f);

jacf = gather(jacf);
% img2_o_f = gather(img2_o_f);
barrierxf = gather(barrierxf);
f_c = gather(f_c);
f_r = gather(f_r);
f_b = gather(f_b);
f_c_temp = gather(f_c_temp);
f_r_temp = gather(f_r_temp);
f_b_temp = gather(f_b_temp);
extf_c = gather(extf_c);
intf_c = gather(intf_c);
d_f_c = gather(d_f_c);

disp(rho);
disp('Final Error:');
disp(sum((img1_o_g(:) - img2_o_f(:)).^2));

% save('out/out.mat');

function plot_profile(img1,img2,row)
hold off;
tmp=img1(row,:);
plot(tmp,'b');
hold on;
tmp=img2(row,:);
plot(tmp,'r');

function img_out=filter_image(img,sigma)
h=fspecial3('gaussian',sigma);
img_out=convn(img,h,'same');

function [dr_img,dc_img,db_img]=grad_img(img)
dr_img=zeros(size(img));
dc_img=zeros(size(img));
db_img=zeros(size(img));

dr_img(2:end-1,2:end-1,2:end-1)=img(3:end,2:end-1,2:end-1)-img(1:end-2,2:end-1,2:end-1); %why is it offset by 2 and not 1?
dc_img(2:end-1,2:end-1,2:end-1)=img(2:end-1,3:end,2:end-1)-img(2:end-1,1:end-2,2:end-1);
db_img(2:end-1,2:end-1,2:end-1)=img(2:end-1,2:end-1,3:end)-img(2:end-1,2:end-1,1:end-2);

% function jac = jacobian(f_c,f_r,f_b)
% jac = ones(size(f_c));
% jac(2:end-1,2:end-1,2:end-1) = (f_c(2:end-1,3:end,2:end-1) - f_c(2:end-1,1:end-2,2:end-1)) .*...
%     ((f_r(3:end,2:end-1,2:end-1) - f_r(1:end-2,2:end-1,2:end-1)) .* (f_b(2:end-1,2:end-1,3:end) - f_b(2:end-1,2:end-1,1:end-2)) - ...
%     (f_r(2:end-1,2:end-1,3:end) - f_r(2:end-1,2:end-1,1:end-2)) .* (f_b(3:end,2:end-1,2:end-1) - f_b(1:end-2,2:end-1,2:end-1))) -...
%     (f_r(2:end-1,3:end,2:end-1) - f_r(2:end-1,1:end-2,2:end-1)) .*...
%     ((f_c(3:end,2:end-1,2:end-1) - f_c(1:end-2,2:end-1,2:end-1)) .* (f_b(2:end-1,2:end-1,3:end) - f_b(2:end-1,2:end-1,1:end-2)) - ...
%     (f_c(2:end-1,2:end-1,3:end) - f_c(2:end-1,2:end-1,1:end-2)) .* (f_b(3:end,2:end-1,2:end-1) - f_b(1:end-2,2:end-1,2:end-1))) +...
%     (f_b(2:end-1,3:end,2:end-1) - f_b(2:end-1,1:end-2,2:end-1)) .*...
%     ((f_c(3:end,2:end-1,2:end-1) - f_c(1:end-2,2:end-1,2:end-1)) .* (f_r(2:end-1,2:end-1,3:end) - f_r(2:end-1,2:end-1,1:end-2)) - ...
%     (f_c(2:end-1,2:end-1,3:end) - f_c(2:end-1,2:end-1,1:end-2)) .* (f_r(3:end,2:end-1,2:end-1) - f_r(1:end-2,2:end-1,2:end-1)));


function h = fspecial3(type,siz)
%FSPECIAL3 Create predefined 3-D filters.
%   H = FSPECIAL3(TYPE,SIZE) creates a 3-dimensional filter H of the
%   specified type and size. Possible values for TYPE are:
%
%     'average'    averaging filter
%     'ellipsoid'  ellipsoidal averaging filter
%     'gaussian'   Gaussian lowpass filter
%     'laplacian'  Laplacian operator
%     'log'        Laplacian of Gaussian filter
%
%   The default SIZE is [5 5 5]. If SIZE is a scalar then H is a 3D cubic
%   filter of dimension SIZE^3.
%
%   H = FSPECIAL3('average',SIZE) returns an averaging filter H of size
%   SIZE. SIZE can be a 3-element vector specifying the dimensions in
%   H or a scalar, in which case H is a cubic array.
%
%   H = FSPECIAL3('ellipsoid',SIZE) returns an ellipsoidal averaging filter.
%
%   H = FSPECIAL3('gaussian',SIZE) returns a centered Gaussian lowpass
%   filter of size SIZE with standard deviations defined as
%   SIZE/(4*sqrt(2*log(2))) so that FWHM equals half filter size
%   (http://en.wikipedia.org/wiki/FWHM). Such a FWHM-dependent standard
%   deviation yields a congruous Gaussian shape (what should be expected
%   for a Gaussian filter!).
%
%   H = FSPECIAL3('laplacian') returns a 3-by-3-by-3 filter approximating
%   the shape of the three-dimensional Laplacian operator. REMARK: the
%   shape of the Laplacian cannot be adjusted. An infinite number of 3D
%   Laplacian could be defined. If you know any simple formulation allowing
%   one to control the shape, please contact me.
%
%   H = FSPECIAL3('log',SIZE) returns a rotationally symmetric Laplacian of
%   Gaussian filter of size SIZE with standard deviation defined as
%   SIZE/(4*sqrt(2*log(2))).
%
%   Class Support
%   -------------
%   H is of class double.
%
%   Example
%   -------
%      I = single(rand(80,40,20));
%      h = fspecial3('gaussian',[9 5 3]); 
%      Inew = imfilter(I,h,'replicate');
%       
%   See also IMFILTER, FSPECIAL.
%
%   -- Damien Garcia -- 2007/08

error(nargchk(1,2,nargin))
type = lower(type);

if nargin==1
        siz = 5;
end

if numel(siz)==1
    siz = round(repmat(siz,1,3));
elseif numel(siz)~=3
    error('Number of elements in SIZ must be 1 or 3')
else
    siz = round(siz(:)');
end

switch type
    
    case 'average'
        h = ones(siz)/prod(siz);
        
    case 'gaussian'
        sig = siz/(4*sqrt(2*log(2)));
        siz   = (siz-1)/2;
        [x,y,z] = ndgrid(-siz(1):siz(1),-siz(2):siz(2),-siz(3):siz(3));
        h = exp(-(x.*x/2/sig(1)^2 + y.*y/2/sig(2)^2 + z.*z/2/sig(3)^2));
        h = h/sum(h(:));
        
    case 'ellipsoid'
        R = siz/2;
        R(R==0) = 1;
        h = ones(siz);
        siz = (siz-1)/2;
        [x,y,z] = ndgrid(-siz(1):siz(1),-siz(2):siz(2),-siz(3):siz(3));
        I = (x.*x/R(1)^2+y.*y/R(2)^2+z.*z/R(3)^2)>1;
        h(I) = 0;
        h = h/sum(h(:));
        
    case 'laplacian'
        h = zeros(3,3,3);
        h(:,:,1) = [0 3 0;3 10 3;0 3 0];
        h(:,:,3) = h(:,:,1);
        h(:,:,2) = [3 10 3;10 -96 10;3 10 3];
        
    case 'log'
        sig = siz/(4*sqrt(2*log(2)));
        siz   = (siz-1)/2;
        [x,y,z] = ndgrid(-siz(1):siz(1),-siz(2):siz(2),-siz(3):siz(3));
        h = exp(-(x.*x/2/sig(1)^2 + y.*y/2/sig(2)^2 + z.*z/2/sig(3)^2));
        h = h/sum(h(:));
        arg = (x.*x/sig(1)^4 + y.*y/sig(2)^4 + z.*z/sig(3)^4 - ...
            (1/sig(1)^2 + 1/sig(2)^2 + 1/sig(3)^2));
        h = arg.*h;
        h = h-sum(h(:))/prod(2*siz+1);
        
    otherwise
        error('Unknown filter type.')
        
end