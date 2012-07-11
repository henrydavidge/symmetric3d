function [f_c,f_r,f_b] = register_images_3D(img1,img2)

%Create two test images
imgSize= 36;
% img1=zeros(36,36,36);
% img1(6:30,6:30,6:30)=1;
% img1=filter_image(img1,5);
% img2=zeros(36,36,36);
% img2(18:30,18:30,18:30)=1;
% img2=filter_image(img2,5);
[x,y,z] = meshgrid(1:imgSize,1:imgSize,1:imgSize);
%Display the images
% figure(1);
% subplot(1,2,1);
% %imshow(mat2gray(img1(:,:,18)));
% slice(x,y,z,img1,[],[],[0,8,18,28,36]);
% subplot(1,2,2);
% %imshow(mat2gray(img2(:,:,18)));
% slice(x,y,z,img2,[],[],[0,8,18,28,36]);

disp('Original Error:');
disp(sum((img1(:) - img2(:)).^2));


%Register the images
%Initialize the registration function %creates lookup tables
%Call the registration function 
tic;
[f_c,f_r,f_b]=register_images(img1,img2,x,y,z,.2,1000);
toc;


   
   


function [f_c,f_r,f_b]=register_images(img1,img2,f_c,f_r,f_b,rho,maxIter)
lambda=0.01;
f_c_init=f_c;
f_r_init=f_r;
f_b_init = f_b;
imgsize=size(f_c);

[x,y,z] = meshgrid(1:36,1:36,1:36);

[dr_img2_orig,dc_img2_orig,db_img2_orig]=grad_img(img2);

for k=1:maxIter,
    %Interpolate image2
    img2_o_f=interp3(img2,f_c,f_r,f_b);
    
    dr_img2=interp3(dr_img2_orig,f_c,f_r,f_b);
    dc_img2=interp3(dc_img2_orig,f_c,f_r,f_b);
    db_img2=interp3(db_img2_orig,f_c,f_r,f_b);

    %external energies
    extf_r=(img1-img2_o_f).*dr_img2; %.* means scalar multiplication (so multiply each element)
    extf_c=(img1-img2_o_f).*dc_img2;
    extf_b=(img1-img2_o_f).*db_img2;
    
    
    %Internal energies
    intf_c=zeros(size(f_c));
    intf_c(2:end-1,2:end-1,2:end-1)= -6*f_c(2:end-1,2:end-1,2:end-1)+ ... %noise reduction?
            f_c(1:end-2,2:end-1,2:end-1)+f_c(3:end,2:end-1,2:end-1)+...
            f_c(2:end-1,1:end-2,2:end-1)+f_c(2:end-1,3:end,2:end-1)+...
            f_c(2:end-1,2:end-1,1:end-2)+f_c(2:end-1,2:end-1,3:end);
    d_f_c=rho*(extf_c+lambda*intf_c);
    d_f_c=min(max(d_f_c,-3),3); %don't want to jump too much in one iteration
    f_c=f_c+d_f_c;
    
    f_c=max(min(f_c,imgsize(1)),1); %edge control


    intf_r=zeros(size(f_r));
    intf_r(2:end-1,2:end-1,2:end-1)= -6*f_r(2:end-1,2:end-1,2:end-1)+ ...
            f_r(1:end-2,2:end-1,2:end-1)+f_r(3:end,2:end-1,2:end-1)+...
            f_r(2:end-1,1:end-2,2:end-1)+f_r(2:end-1,3:end,2:end-1)+...
            f_r(2:end-1,2:end-1,1:end-2)+f_r(2:end-1,2:end-1,3:end);
    d_f_r=rho*(extf_r+lambda*intf_r);
    d_f_r=min(max(d_f_r,-3),3);
    f_r=f_r+d_f_r;
    
    f_r=max(min(f_r,imgsize(1)),1);
    
    intf_b=zeros(size(f_b));
    intf_b(2:end-1,2:end-1,2:end-1) = -6*f_b(2:end-1,2:end-1,2:end-1)+...
        f_b(1:end-2,2:end-1,2:end-1)+f_b(3:end,2:end-1,2:end-1)+...
        f_b(2:end-1,1:end-2,2:end-1)+f_b(2:end-1,3:end,2:end-1)+...
        f_b(2:end-1,2:end-1,1:end-2)+f_b(2:end-1,2:end-1,3:end);
    d_f_b=rho*(extf_b+lambda*intf_b);
    d_f_b=min(max(d_f_b,-3),3);
    f_b=f_b+d_f_b;
    
    f_b=max(min(f_b,imgsize(1)),1);


%     figure(2);
%     subplot(1,2,1);
%     slice(x,y,z,img2_o_f,[],[],[0,8,18,28,36]);
%     %imshow(mat2gray(img2_o_f(:,:,12)));
%     subplot(1,2,2);
%     tmp=-img1+img2_o_f;
%     tmp(1,1)=1;
%     slice(x,y,z,tmp,[],[],[0,8,18,28,36]);
%     %imshow(mat2gray(tmp(:,:,12)));
%     
%     figure(3);
%     plot_profile(img1,img2_o_f,18);
    
    
end
disp('Final Error:');
disp(sum((img1(:) - img2_o_f(:)).^2));

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