import torch
import os

def mkdirp(path):
    try:
        os.makedirs(path)
    except OSError:
        pass
        
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def disp_time(time_secs):
	
	hrs = int(time_secs/3600)
	mins = int((time_secs - hrs*3600)/60)
	secs = time_secs - hrs*3600 - mins*60
	disp_str = '%d hrs, %d mins, %d secs' % (hrs, mins, secs)
	return disp_str
