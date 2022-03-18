from src import jsw_reg as reg, jsw_seg as seg
import os
"""
The main script to run both methods
"""
# Regression based
print('##### REGRESSION BASED #####')
# train ML model
# reg.train_jsw_reg_model(oa=False, crop_type='subregion')

# run method
kl = 4
reg.run_jsw_reg_subregions(kl, crop_type='center')
reg.run_jsw_reg_subregions(kl, crop_type='subregion')


# Segmentation based
print('##### SEGMENTATION BASED #####')
# run method
seg.run_jsw_seg(kl)

