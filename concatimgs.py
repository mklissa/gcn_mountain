from PIL import Image as im
import numpy as np
import pdb
# names = ['800','nowall','nowallfeatures','wall']
# appendix = ['','_rounded']

# for name in names:

# 	imgs=[]
# 	for app in appendix:
# 		imgs.append(np.array(im.open("{}{}.png".format(name,app))))
	
# 	end= np.hstack(imgs)
# 	imgs_comb = im.fromarray( end)
# 	imgs_comb.save( '{}_concat.png'.format(name) )   



# imgs=[]
# for name in names:
# 	imgs.append(np.array(im.open("{}_concat.png".format(name))))

# end= np.vstack(imgs)
# imgs_comb = im.fromarray( end)
# imgs_comb.save( 'concatall.png'.format(name) )   



appendix= ['vpreds0/vpred367','vpreds/vpred28']
imgs=[]
for app in appendix:
	imgs.append(np.array(im.open("{}.png".format(app))))
# pdb.set_trace()
end= np.hstack(imgs)
imgs_comb = im.fromarray( end)
imgs_comb.save( 'carvf.png' )   

