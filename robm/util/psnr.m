function [psnrval]= psnr( range, mse)
psnrval = 20*log10( range/sqrt(mse));