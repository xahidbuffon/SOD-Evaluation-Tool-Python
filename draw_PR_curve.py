"""
# > Script for PR-curve evaluation of SOD methods on benchmark datasets 
    * Metrics: standard PR evaluation in [0, 255] 
      // download the evaluation data for testing
    * Consistent with the Matlab version 
      // details: https://github.com/wenguanwang/SODsurvey    
# Maintainer: https://github.com/xahidbuffon
"""
import os
import ntpath
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
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

 
plot_vals = {}
### for every model (per dataset)
for method in methods_info.keys():
    out_dir = os.path.join(eval_dir, dataset, method)
    out_paths  = [os.path.join(out_dir, f) for f in sorted(os.listdir(out_dir))]

    # evaluation pipeline
    all_p, all_r = [], []
    for i in range(len(gt_paths)):
        gt = read_mask(gt_paths[i],  res=eval_res) #[0, 255]
        im_name = ntpath.basename(gt_paths[i]).split('.')[0]
        smap_name = os.path.join(out_dir, im_name+methods_info[method]) 
        if smap_name not in out_paths:
            continue
        # get precision and recall values (255 bins of thresholds)
        smap = read_mask(smap_name, res=eval_res) # [0, 255]
        Ps, Rs, _ = get_PR_uint8(smap, gt)
        all_p.append(Ps); all_r.append(Rs)
    # save the PR curve values per {"method": (R, P)}
    Ps = np.mean(np.array(all_p), 0)
    Rs = np.mean(np.array(all_r), 0)
    plot_vals[method] = (Rs, Ps) 

## plot the PR curves
plt.clf()
colors = 'rkbmc'; ticks = ['-', '--']
for i, m in enumerate(methods_info.keys()):
    x, y = plot_vals[m]
    marker = colors[i%len(colors)] + ticks[i%2] 
    plt.plot(x, y, marker, linewidth=2, label=m)

plt.grid(True)
_font_size_ = 16
plt.title(dataset, fontsize=_font_size_+2)
plt.xlim([0.55, 1.0]); #plt.ylim([0.0, 1.0])
plt.xlabel("Recall", fontsize=_font_size_); plt.xticks(fontsize=_font_size_-4)
plt.ylabel("Precision", fontsize=_font_size_); plt.yticks(fontsize=_font_size_-4)
plt.legend(methods_info.keys(), loc='lower left', fontsize=_font_size_-2, framealpha=0.75)
plt.savefig('pr.png', bbox_inches='tight')
plt.show()


