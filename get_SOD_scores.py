"""
# > Script for quantitative evaliation of SOD methods on benchmark datasets 
    * Metrics: MAE, S-Measure, and F-Measure (per method per dataset)
      // download the evaluation data for testing
    * Consistent with the Matlab version 
      // details: https://github.com/wenguanwang/SODsurvey    
# Maintainer: https://github.com/xahidbuffon
"""
import os
import ntpath
import numpy as np
from PIL import Image
from measures.MAE import get_MAE
from measures.S_Measure import get_SMeasure
from measures.F_Measure import get_PR_uint8, get_wFMeasure

eval_dir = 'eval_data/terrestrial/' 
dataset  = 'PASCAL-S'
eval_res = (224, 224)
# {method, file extension}
methods_info = {'AFNet': '.png', 
                'ASNet':'.png', 
                'BASNet':'.png', 
                'CPD':'.png', 
                'MLMSNet':'.jpg', 
                'PAGE-Net':'.png',
                'PAGRN18':'.png',
                'PiCANet':'.png',
                } 

#eval_dir = 'eval_data/underwater/'  
#dataset  = 'USOD' # 'UFO-120' / 'SUIM'/ 'USOD'
#eval_res = (256, 256)
## {method, file extension}
#methods_info = {
#                'Deep_SESR': '.png',
#                'LSM': '.png',   
#                'SUIM_Net': '.bmp', 
#                'QDWD': '.png', 
#                'SVAM-Net': '.png',
#                'SVAM-Net_light': '.png',
#                'ASNet': '.jpg',
#                'BASNet': '.png',
#                'CPD': '.png',
#                'PAGE-Net': '.png'
#                } 

gt_dir = os.path.join(eval_dir, dataset, 'GT')
gt_paths  = [os.path.join(gt_dir, f) for f in sorted(os.listdir(gt_dir))]


def read_mask(path, res=(224, 224)):
    # read image and return array (0, 255)
    img = Image.open(path).resize(res)
    return np.array(img.convert("L"))


def read_and_scale_mask(path, res=(224, 224), thr=0.5):
    """
       - Get image from given path and reshape to res
       - Return as array  
    """
    img = read_mask(path, res)/255.0
    return (img > thr).astype(np.int32)


### for every model (per dataset)
for method in methods_info.keys():
    out_dir = os.path.join(eval_dir, dataset, method)
    out_paths  = [os.path.join(out_dir, f) for f in sorted(os.listdir(out_dir))]

    # evaluation pipeline
    MAEs, SMeasures, F1s = [], [], []
    all_p, all_r, all_iou = [], [], []
    for i in range(len(gt_paths)):
        gt = read_and_scale_mask(gt_paths[i],  res=eval_res) #[0/1]
        im_name = ntpath.basename(gt_paths[i]).split('.')[0]
        smap_name = os.path.join(out_dir, im_name+methods_info[method]) 
        if smap_name not in out_paths:
            continue

        # MAE and S_Mesure
        smap = read_and_scale_mask(smap_name, res=eval_res) #[0/1]
        MAEs.append(get_MAE(smap, gt))
        SMeasures.append(get_SMeasure(smap, gt))

        # Weighted F_mesure_max (255 bins of thresholds)
        smap = read_mask(smap_name, res=eval_res) # [0, 255]
        Ps, Rs, ious = get_PR_uint8(smap, gt)
        F1s.append(get_wFMeasure(Ps, Rs))
        all_p.append(Ps) 
        all_r.append(Rs) 
        all_iou.append(ious)
    F1s = np.mean(np.array(F1s), 0)
    F1s_max = np.max(F1s)

    # results
    print ("\n{0} on {1} ({2} images):".format(method, dataset, len(MAEs)))
    print ("-------------------------------------------------")
    print ("Mean MAE: {0}".format(np.round(np.mean(MAEs), 4)))
    print ("Mean S-Measure: {0}".format(np.round(np.mean(SMeasures), 4)))
    print ("Mean F-Measure: {0}".format(np.round(F1s_max, 4)))

