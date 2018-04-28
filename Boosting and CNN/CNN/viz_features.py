from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random
import cv2
import IPython
import numpy as np



class Viz_Feat(object):


    def __init__(self,val_data,train_data, class_labels,sess):

        self.val_data = val_data
        self.train_data = train_data
        self.CLASS_LABELS = class_labels
        self.sess = sess





    def vizualize_features(self,net):

        images = [0,10,100]
        '''
        Compute the response map for the index images
        '''
        
        for i in images:

            data_val = self.val_data[i]
            val_eval = np.zeros([1, data_val['features'].shape[0], data_val['features'].shape[1], data_val['features'].shape[2]])
            val_eval[0,:,:,:] = data_val['features']
            val_label = np.zeros([1,len(self.CLASS_LABELS)]) 
            val_label[0,:] = data_val['label']
            response_map = self.sess.run(net.response_map, feed_dict={net.images: val_eval, net.labels: val_label})
            img_name = str(i)+'th_image.png'
            cv2.imwrite(img_name, data_val['c_img'])
            for j in range(5):
                img = self.revert_image(response_map[0,:,:,j])
                img_name = str(i)+'th_image'+'_'+str(j)+'th_filter.png'
                cv2.imwrite(img_name, img)



    def revert_image(self,img):
        '''
        Used to revert images back to a form that can be easily visualized
        '''

        img = (img+1.0)/2.0*255.0

        img = np.array(img,dtype=int)

        blank_img = np.zeros([img.shape[0],img.shape[1],3])

        blank_img[:,:,0] = img
        blank_img[:,:,1] = img
        blank_img[:,:,2] = img

        img = blank_img.astype("uint8")

        return img

        




