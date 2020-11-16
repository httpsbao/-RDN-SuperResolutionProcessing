import numpy as np
from PIL import Image
from ISR.models import RDN

img = Image.open('data/flower.png')
lr_img = np.array(img)

#Large RDN model
def fun1():
    rdn = RDN(arch_params={'C':6, 'D':20, 'G':64, 'G0':64, 'x':2})
    rdn.model.load_weights('weights/rdn-C6-D20-G64-G064-x2_PSNR_epoch086.hdf5')
    #Run prediction
    sr_img = rdn.predict(lr_img)
    Image.fromarray(sr_img)
    ans=Image.fromarray(sr_img)
    ss="out1.png"
    ans.save(ss)

#Small RDN model
def fun2():
    rdn = RDN(arch_params={'C':3, 'D':10, 'G':64, 'G0':64, 'x':2})
    rdn.model.load_weights('weights/rdn-C3-D10-G64-G064-x2_PSNR_epoch134.hdf5')
    #Run prediction
    sr_img = rdn.predict(lr_img)
    Image.fromarray(sr_img)
    ans=Image.fromarray(sr_img)
    ss="out2.png"
    ans.save(ss)

#Large RDN noise cancelling, detail enhancing model
def fun3():
    rdn = RDN(arch_params={'C':6, 'D':20, 'G':64, 'G0':64, 'x':2})
    rdn.model.load_weights('weights/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5')
    #Run prediction
    sr_img = rdn.predict(lr_img)
    Image.fromarray(sr_img)
    ans=Image.fromarray(sr_img)
    ss="out3.png"
    ans.save(ss)

number=input('Please input the number of function you want to run:')
print('We will run fun' + number + ':')
if(number=='1'):
    print(number)
    fun1()
elif(number=='2'):
    fun2()
elif(number=='3'):
    fun3()
