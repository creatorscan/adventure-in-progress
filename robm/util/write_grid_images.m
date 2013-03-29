%{
===========================================================================
Code provided by Yichuan (Charlie) Tang
Permission is granted for anyone to copy, use, modify, or distribute this
program and accompanying programs and documents for any purpose, provided
this copyright notice is retained and prominently displayed, along with
a note saying that the original programs are available from our
web page.
The programs and documents are distributed without any warranty, express or
implied.  As the programs were written for research purposes only, they 
have not been tested to the degree that would be advisable in any important
application.  All use of these programs is entirely at the user's own risk.
===========================================================================

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
CT 
PURPOSE:
INPUT: Data  should be n x d, where n is the number of images, d is the
		 serialized image data, row major. imagedim is nRows xnCols of 
			image size
		griddim: images in big image
		borderwidth: number of pixels between images
		borderval: [0 to 1] value of border between images                
OUTPUT:
NOTES:  This function assumes that the images are row by row serialization 
		of Data(i,:) into a 2D image
		RGB is supported!!
TESTED:
CHANGELOG:
TODO:       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}

function [image] = write_grid_images(Data, imagedim, griddim, borderwidth, borderval)


image = borderval*ones(griddim(1)*imagedim(1)+borderwidth*(griddim(1)+1), ...
                griddim(2)*imagedim(2)+borderwidth*(griddim(2)+1),size(Data,3));
            
for i = 1:griddim(1)    
    for j = 1:griddim(2)
                
        ival = (i-1)*(imagedim(1)+borderwidth)+1+borderwidth;
        jval = (j-1)*(imagedim(2)+borderwidth)+1+borderwidth;
               
        im_2d = reshape(Data( (i-1)*griddim(2)+j, :,:), imagedim(2) , imagedim(1) ,size(Data,3));        
        %the imagedim(2) first hen imagedim(1) is because we do a
        %transpose at the end
        
        if length(size(im_2d)) == 3
            im_2d(:,:,1) = im_2d(:,:,1)';
            im_2d(:,:,2) = im_2d(:,:,2)';
            im_2d(:,:,3) = im_2d(:,:,3)';
        else
           im_2d = im_2d'; 
        end
        
        image(ival:ival+imagedim(1)-1, jval:jval+imagedim(2)-1,:) = im_2d;      
    end
end


