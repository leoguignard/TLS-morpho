from tifffile import imread
from Computing_metrics import compute_metrics2D, compute_metrics3D

# 2D computation
bin_im = imread('data/im_mask2D.tif')
dist_trsf = imread('data/dist_trsf_im2D.tif')
AP_pos = ([876, 1016], [173, 616])
vs = 0.82
compute_metrics2D(bin_im, dist_trsf, AP_pos=AP_pos, vs=vs)

# 3D computation
bin_im = imread('data/im_mask3D.tif')
dist_trsf = imread('data/dist_trsf_im3D.tif')
vs = 8.3
compute_metrics3D(bin_im, dist_trsf, vs=vs)