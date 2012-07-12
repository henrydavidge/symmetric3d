function [f_c,f_r,f_b,g_c,g_r,g_b] = register_images_3D_symmetricb(img1,img2,rho,lambda,lambda2,maxIter)

imgSize = size(img1);
rows = imgSize(1);
cols = imgSize(2);
beams = imgSize(3);
rho_init = rho;

disp('Initial Error');
disp(sum((img1(:) - img2(:)).^2));
tic;



%Register the images
%Initialize the registration function
[f_c,f_r,f_b] = meshgrid(1:cols,1:rows,1:beams);
[g_c,g_r,g_b] = meshgrid(1:cols,1:rows,1:beams);
%Call the registration function 
f_c_init=f_c;
f_r_init=f_r;
f_b_init = f_b;

g_c_init=g_c;
g_r_init=g_r;
g_b_init = g_b;

[dr_img1_orig,dc_img1_orig,db_img1_orig]=grad_img(img1);
[dr_img2_orig,dc_img2_orig,db_img2_orig]=grad_img(img2);

% compile cuda file%
if (exist('symmetric3d.ptx','file') ~= 2)
   if (ispc)
       [status, result] = system('"C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\vcvarsall.bat" & nvcc -ptx symmetric3d.cu','-echo');
   else
       [status, result] = system('nvcc -ptx symmetric3d.cu','-echo');
   end
end

%calculate block and grid sizes
block = [8,8,8];
while (mod(rows,block(2)) ~= 0)
    block(2) = block(2)/2;
    block(3) = block(3) * 2;
end
grid = [ceil(cols/block(1)), ceil(rows/block(2)) * ceil(beams/block(3))];
disp(grid);

%initialize kernels%
interp = parallel.gpu.CUDAKernel('symmetric3d.ptx', 'symmetric3d.cu', 'interp');
interp.ThreadBlockSize = [6,6,6];
interp.GridSize = [6,36];

extf = parallel.gpu.CUDAKernel('symmetric3d.ptx', 'symmetric3d.cu', 'extf');
extf.ThreadBlockSize = block;
extf.GridSize = grid;

jacPartialsAndBarrier = parallel.gpu.CUDAKernel('symmetric3d.ptx', 'symmetric3d.cu', 'jacPartials');
jacPartialsAndBarrier.ThreadBlockSize = block;
jacPartialsAndBarrier.GridSize = grid;

intf = parallel.gpu.CUDAKernel('symmetric3d.ptx', 'symmetric3d.cu', 'intf');
intf.ThreadBlockSize = block;
intf.GridSize = grid;

jacobian = parallel.gpu.CUDAKernel('symmetric3d.ptx', 'symmetric3d.cu', 'jacobian');
jacobian.ThreadBlockSize = block;
jacobian.GridSize = grid;

d_f = parallel.gpu.CUDAKernel('symmetric3d.ptx', 'symmetric3d.cu', 'd_f');
d_f.ThreadBlockSize = block;
d_f.GridSize = grid;

%initialize variables on the gpu
img2_o_f = parallel.gpu.GPUArray.zeros(rows, cols, beams);
jacf = parallel.gpu.GPUArray.ones(rows, cols, beams);

img1_o_g = parallel.gpu.GPUArray.zeros(rows, cols, beams);
jacg = parallel.gpu.GPUArray.ones(rows, cols, beams);

f_c = gpuArray(f_c);
dc_img2 = parallel.gpu.GPUArray.zeros(rows, cols, beams);

d_f_c = parallel.gpu.GPUArray.zeros(rows, cols, beams);

i_m_1xf = parallel.gpu.GPUArray.zeros(rows, cols, beams);
i_p_1xf = parallel.gpu.GPUArray.zeros(rows, cols, beams);
j_m_1xf = parallel.gpu.GPUArray.zeros(rows, cols, beams);
j_p_1xf = parallel.gpu.GPUArray.zeros(rows, cols, beams);
k_m_1xf = parallel.gpu.GPUArray.zeros(rows, cols, beams);
k_p_1xf = parallel.gpu.GPUArray.zeros(rows, cols, beams);

barrierxf = parallel.gpu.GPUArray.zeros(rows, cols, beams);

extf_c = parallel.gpu.GPUArray.zeros(rows, cols, beams);
intf_c = parallel.gpu.GPUArray.zeros(rows, cols, beams);

f_r = gpuArray(f_r);
dr_img2 = parallel.gpu.GPUArray.zeros(rows, cols, beams);

d_f_r = parallel.gpu.GPUArray.zeros(rows, cols, beams);

i_m_1yf = parallel.gpu.GPUArray.zeros(rows, cols, beams);
i_p_1yf = parallel.gpu.GPUArray.zeros(rows, cols, beams);
j_m_1yf = parallel.gpu.GPUArray.zeros(rows, cols, beams);
j_p_1yf = parallel.gpu.GPUArray.zeros(rows, cols, beams);
k_m_1yf = parallel.gpu.GPUArray.zeros(rows, cols, beams);
k_p_1yf = parallel.gpu.GPUArray.zeros(rows, cols, beams);

barrieryf = parallel.gpu.GPUArray.zeros(rows, cols, beams);

extf_r = parallel.gpu.GPUArray.zeros(rows, cols, beams);
intf_r = parallel.gpu.GPUArray.zeros(rows, cols, beams);

f_b = gpuArray(f_b);
db_img2 = parallel.gpu.GPUArray.zeros(rows, cols, beams);

d_f_b = parallel.gpu.GPUArray.zeros(rows, cols, beams);

i_m_1zf = parallel.gpu.GPUArray.zeros(rows, cols, beams);
i_p_1zf = parallel.gpu.GPUArray.zeros(rows, cols, beams);
j_m_1zf = parallel.gpu.GPUArray.zeros(rows, cols, beams);
j_p_1zf = parallel.gpu.GPUArray.zeros(rows, cols, beams);
k_m_1zf = parallel.gpu.GPUArray.zeros(rows, cols, beams);
k_p_1zf = parallel.gpu.GPUArray.zeros(rows, cols, beams);

barrierzf = parallel.gpu.GPUArray.zeros(rows, cols, beams);

extf_b = parallel.gpu.GPUArray.zeros(rows, cols, beams);
intf_b = parallel.gpu.GPUArray.zeros(rows, cols, beams);

g_c = gpuArray(g_c);
dc_img1 = parallel.gpu.GPUArray.zeros(rows, cols, beams);

d_g_c = parallel.gpu.GPUArray.zeros(rows, cols, beams);

i_m_1xg = parallel.gpu.GPUArray.zeros(rows, cols, beams);
i_p_1xg = parallel.gpu.GPUArray.zeros(rows, cols, beams);
j_m_1xg = parallel.gpu.GPUArray.zeros(rows, cols, beams);
j_p_1xg = parallel.gpu.GPUArray.zeros(rows, cols, beams);
k_m_1xg = parallel.gpu.GPUArray.zeros(rows, cols, beams);
k_p_1xg = parallel.gpu.GPUArray.zeros(rows, cols, beams);

barrierxg = parallel.gpu.GPUArray.zeros(rows, cols, beams);

extg_c = parallel.gpu.GPUArray.zeros(rows, cols, beams);
intg_c = parallel.gpu.GPUArray.zeros(rows, cols, beams);

g_r = gpuArray(g_r);
dr_img1 = parallel.gpu.GPUArray.zeros(rows, cols, beams);

d_g_r = parallel.gpu.GPUArray.zeros(rows, cols, beams);

i_m_1yg = parallel.gpu.GPUArray.zeros(rows, cols, beams);
i_p_1yg = parallel.gpu.GPUArray.zeros(rows, cols, beams);
j_m_1yg = parallel.gpu.GPUArray.zeros(rows, cols, beams);
j_p_1yg = parallel.gpu.GPUArray.zeros(rows, cols, beams);
k_m_1yg = parallel.gpu.GPUArray.zeros(rows, cols, beams);
k_p_1yg = parallel.gpu.GPUArray.zeros(rows, cols, beams);

barrieryg = parallel.gpu.GPUArray.zeros(rows, cols, beams);

extg_r = parallel.gpu.GPUArray.zeros(rows, cols, beams);
intg_r = parallel.gpu.GPUArray.zeros(rows, cols, beams);

g_b = gpuArray(g_b);
db_img1 = parallel.gpu.GPUArray.zeros(rows, cols, beams);

d_g_b = parallel.gpu.GPUArray.zeros(rows, cols, beams);

i_m_1zg = parallel.gpu.GPUArray.zeros(rows, cols, beams);
i_p_1zg = parallel.gpu.GPUArray.zeros(rows, cols, beams);
j_m_1zg = parallel.gpu.GPUArray.zeros(rows, cols, beams);
j_p_1zg = parallel.gpu.GPUArray.zeros(rows, cols, beams);
k_m_1zg = parallel.gpu.GPUArray.zeros(rows, cols, beams);
k_p_1zg = parallel.gpu.GPUArray.zeros(rows, cols, beams);

barrierzg = parallel.gpu.GPUArray.zeros(rows, cols, beams);

extg_b = parallel.gpu.GPUArray.zeros(rows, cols, beams);
intg_b = parallel.gpu.GPUArray.zeros(rows, cols, beams);

for k=1:maxIter,
    if (mod(k,250) == 0)
        rho = rho_init;
        disp(k);
        toc;
    end
%     %Interpolate image1
    img1_o_g = feval(interp, img1_o_g, g_c, g_r, g_b, img1, rows, cols, beams);
    dr_img1 = feval(interp, dr_img1, g_c, g_r, g_b, dr_img1_orig, rows, cols, beams);
    dc_img1 = feval(interp, dc_img1, g_c, g_r, g_b, dc_img1_orig, rows, cols, beams);
    db_img1 = feval(interp, db_img1, g_c, g_r, g_b, db_img1_orig, rows, cols, beams);
    
    %Interpolate image2
    img2_o_f = feval(interp, img2_o_f, f_c, f_r, f_b, img2, rows, cols, beams);
    dr_img2 = feval(interp, dr_img2, f_c, f_r, f_b, dr_img2_orig, rows, cols, beams);
    dc_img2 = feval(interp, dc_img2, f_c, f_r, f_b, dc_img2_orig, rows, cols, beams);
    db_img2 = feval(interp, db_img2, f_c, f_r, f_b, db_img2_orig, rows, cols, beams);

    %external energies
    extg_r = -feval(extf, extg_r, img1_o_g, img2_o_f, dr_img1, rows, cols, beams); 
    extg_c = -feval(extf, extg_c, img1_o_g, img2_o_f, dc_img1, rows, cols, beams);
    extg_b = -feval(extf, extg_b, img1_o_g, img2_o_f, db_img1, rows, cols, beams);
    
    extf_r = feval(extf, extf_r, img1_o_g, img2_o_f, dr_img2, rows, cols, beams); 
    extf_c = feval(extf, extf_c, img1_o_g, img2_o_f, dc_img2, rows, cols, beams);
    extf_b = feval(extf, extf_b, img1_o_g, img2_o_f, db_img2, rows, cols, beams);
    
    
    %calculating partials of the jacobian and th barrier function
    [i_m_1xg, i_p_1xg, j_m_1xg, j_p_1xg, k_m_1xg, k_p_1xg, barrierxg] = feval(jacPartialsAndBarrier,...
        i_m_1xg, i_p_1xg, j_m_1xg, j_p_1xg, k_m_1xg, k_p_1xg, barrierxg,...
        jacg, g_r, g_b, img1_o_g, img2_o_f, rows, cols, beams, 1);
    
    [i_m_1yg, i_p_1yg, j_m_1yg, j_p_1yg, k_m_1yg, k_p_1yg, barrieryg] = feval(jacPartialsAndBarrier,...
        i_m_1yg, i_p_1yg, j_m_1yg, j_p_1yg, k_m_1yg, k_p_1yg, barrieryg,...
        jacg, g_c, g_b, img1_o_g, img2_o_f, rows, cols, beams, -1);
    
    [i_m_1zg, i_p_1zg, j_m_1zg, j_p_1zg, k_m_1zg, k_p_1zg, barrierzg] = feval(jacPartialsAndBarrier,...
        i_m_1zg, i_p_1zg, j_m_1zg, j_p_1zg, k_m_1zg, k_p_1zg, barrierzg,...
        jacg, g_c, g_r, img1_o_g, img2_o_f, rows, cols, beams, 1);
    
    [i_m_1xf, i_p_1xf, j_m_1xf, j_p_1xf, k_m_1xf, k_p_1xf, barrierxf] = feval(jacPartialsAndBarrier,...
        i_m_1xf, i_p_1xf, j_m_1xf, j_p_1xf, k_m_1xf, k_p_1xf, barrierxf,...
        jacf, f_r, f_b, img1_o_g, img2_o_f, rows, cols, beams, 1);
    
    [i_m_1yf, i_p_1yf, j_m_1yf, j_p_1yf, k_m_1yf, k_p_1yf, barrieryf] = feval(jacPartialsAndBarrier,...
        i_m_1yf, i_p_1yf, j_m_1yf, j_p_1yf, k_m_1yf, k_p_1yf, barrieryf,...
        jacf, f_c, f_b, img1_o_g, img2_o_f, rows, cols, beams, -1);
    
    [i_m_1zf, i_p_1zf, j_m_1zf, j_p_1zf, k_m_1zf, k_p_1zf, barrierzf] = feval(jacPartialsAndBarrier,...
        i_m_1zf, i_p_1zf, j_m_1zf, j_p_1zf, k_m_1zf, k_p_1zf, barrierzf,...
        jacf, f_c, f_r, img1_o_g, img2_o_f, rows, cols, beams, 1);
    
    
    
    
                    
    %Internal energies
    intg_c = feval(intf, intg_c, g_c, rows, cols, beams);
        
    intg_r = feval(intf, intg_r, g_r, rows, cols, beams);
        
    intg_b = feval(intf, intg_b, g_b, rows, cols, beams);
    
    intf_c = feval(intf, intf_c, f_c, rows, cols, beams);
        
    intf_r = feval(intf, intf_r, f_r, rows, cols, beams);
        
    intf_b = feval(intf, intf_b, f_b, rows, cols, beams);
    
    while(1)
        d_g_c = feval(d_f,d_g_c,jacf,jacg,extg_c,i_m_1xg,i_p_1xg,j_m_1xg,j_p_1xg,k_m_1xg,k_p_1xg,intg_c,barrierxg,rho,lambda,lambda2,rows, cols, beams);
        d_f_c=min(max(d_f_c,-3),3); %don't want to jump too much in one iteration
        g_c_temp=g_c+d_g_c;
        
        g_c_temp = max(min(g_c_temp, cols), 1);
        
        
        
        
        d_g_r = feval(d_f,d_g_r,jacf,jacg,extg_r,i_m_1yg,i_p_1yg,j_m_1yg,j_p_1yg,k_m_1yg,k_p_1yg,intg_r,barrieryg,rho,lambda,lambda2,rows, cols, beams);
        d_f_r=min(max(d_f_r,-3),3);
        g_r_temp=g_r+d_g_r;
        
        g_r_temp = max(min(g_r_temp, rows), 1);
        
        d_g_b = feval(d_f,d_g_b,jacf,jacg,extg_b,i_m_1zg,i_p_1zg,j_m_1zg,j_p_1zg,k_m_1zg,k_p_1zg,intg_b,barrierzg,rho,lambda,lambda2,rows, cols, beams);
        d_f_b=min(max(d_f_b,-3),3);
        g_b_temp=g_b+d_g_b;
        
        g_b_temp = max(min(g_b_temp, beams), 1);
        
        
        d_f_c = feval(d_f,d_f_c,jacf,jacg,extf_c,i_m_1xf,i_p_1xf,j_m_1xf,j_p_1xf,k_m_1xf,k_p_1xf,intf_c,barrierxf,rho,lambda,lambda2,rows, cols, beams);
        d_f_c=min(max(d_f_c,-3),3); %don't want to jump too much in one iteration
        f_c_temp=f_c+d_f_c;
        
        f_c_temp = max(min(f_c_temp, cols), 1);
        
        
        
        
        d_f_r = feval(d_f,d_f_r,jacf,jacg,extf_r,i_m_1yf,i_p_1yf,j_m_1yf,j_p_1yf,k_m_1yf,k_p_1yf,intf_r,barrieryf,rho,lambda,lambda2,rows, cols, beams);
        d_f_r=min(max(d_f_r,-3),3);
        f_r_temp=f_r+d_f_r;
        
        f_r_temp = max(min(f_r_temp, rows), 1);
        
        d_f_b = feval(d_f,d_f_b,jacf,jacg,extf_b,i_m_1zf,i_p_1zf,j_m_1zf,j_p_1zf,k_m_1zf,k_p_1zf,intf_b,barrierzf,rho,lambda,lambda2,rows, cols, beams);
        d_f_b=min(max(d_f_b,-3),3);
        f_b_temp=f_b+d_f_b;
        
        f_b_temp = max(min(f_b_temp, beams), 1);
        
        jacf_temp = feval(jacobian, jacf, f_c_temp, f_r_temp, f_b_temp, rows, cols, beams);
        jacg_temp = feval(jacobian, jacg, g_c_temp, g_r_temp, g_b_temp, rows, cols, beams);
        
        jacfmin = min(jacf_temp(:));
        jacgmin = min(jacg_temp(:));

        if (jacfmin >= 0 && jacgmin >= 0)
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

% %Write final Spider files
% img1_o_g = gather(img1_o_g);
% img2_o_f = gather(img2_o_f);
% 
% jacf = gather(jacf);
% img2_o_f = gather(img2_o_f);
% barrierxf = gather(barrierxf);
% f_c = gather(f_c);
% f_r = gather(f_r);
% f_b = gather(f_b);
% f_c_temp = gather(f_c_temp);
% f_r_temp = gather(f_r_temp);
% f_b_temp = gather(f_b_temp);
% extf_c = gather(extf_c);
% intf_c = gather(intf_c);
% d_f_c = gather(d_f_c);

disp(rho);
disp('Final Error:');
disp(sum((img1_o_g(:) - img2_o_f(:)).^2));

save('out/out.mat', 'img1_o_g','img2_o_f','f_c','f_r','f_b','g_c','g_r','g_b');

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


