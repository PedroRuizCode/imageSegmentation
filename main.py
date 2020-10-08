'''         Image processing and computer vision
              Alejandra Avendaño y Pedro Ruiz
               Electronic engineering students
              Pontificia Universidad Javeriana
                      Bogotá - 2020
'''
import numpy as np #import numpy library
import os #import os library
import cv2 #import openCV library
import sys #import sys library
import matplotlib.pyplot as plt #import matplotlib library
from sklearn.mixture import GaussianMixture as GMM #import GaussianMixture library
from sklearn.cluster import KMeans #import KMeans library
from sklearn.utils import shuffle #import shuffle library
from time import time #import time library

def recreate_image (centers, labels, rows, cols):#create a function
    d = centers.shape[1]
    image_clusters = np.zeros((rows, cols, d)) #create zeros matrix
    label_idx = 0
    for i in range(rows):
        for j in range(cols):
            image_clusters[i][j] = centers[labels[label_idx]]
            label_idx += 1
    return image_clusters

if __name__ == '__main__':
    method = input("Which method do you want to choose? Options are: kmeans & gmm ")#Ask for segmentation method
    print ("The method you selected is ", method, '\n')
    f_dist = np.zeros(10, float)

    for n_cluster in range(10):#segment with 1 to 10 clusters
        print('Clustering for ', n_cluster + 1, 'clusters\n') #print string
        n_colors = n_cluster + 1

        path = sys.argv[1] #Path of images
        image_name = sys.argv[2] #name of the image
        path_file = os.path.join(path, image_name)
        image = cv2.imread(path_file) #Upload the image

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #convert image from BGR to RGB
        image = np.array(image, dtype=np.float64) / 255 #Normalize data from 0 to 1
        rows, cols, ch = image.shape
        assert ch == 3
        image_array = np.reshape(image, (rows * cols, ch)) #Reorder data
        print("Fitting model on a small sub-sample of the data") #print string
        t0 = time() #Save actual time
        image_array_sample = shuffle(image_array, random_state=0)[:10000] #Sample original image
        if method == 'gmm':
            model = GMM(n_components=n_colors).fit(image_array_sample) #define the model
        else:
            model = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample) #define the model
        print("Done in %0.3fs." % (time() - t0)) #print time

        # Get labels for all points
        t0 = time() #Save actual time
        if method == 'gmm':
            print("Predicting color indices on the full image (GMM)") #print string
            labels = model.predict(image_array) #Assign a label to each pixel
            centers = model.means_ #define the centers of the clusters
        else:
            print("Predicting color indices on the full image (Kmeans)") #print string
            labels = model.predict(image_array) #Assign a label to each pixel
            centers = model.cluster_centers_ #define the centers of the clusters
        print("Done in %0.3fs." % (time() - t0)) #print time

        dist = np.zeros(n_colors, float)
        for i in range(labels.shape[0]):
            cl = labels[i]
            #Calculate euclidean distance
            dist[cl] += np.sqrt(((image_array[i,0]-centers[cl,0])**2)+((image_array[i,1]-centers[cl,1])**2)+((image_array[i,2]-centers[cl,2])**2))
        f_dist[n_cluster] = np.sum(dist)#Add all distances
        print('The value of the intra cluster distance is ', f_dist[n_cluster]) #print distance
        print('\n')

        #show resulting image
        plt.figure(n_cluster + 1)
        plt.clf()
        plt.axis('off')
        plt.title('Quantized image ({} colors, method={})'.format(n_colors, method))
        plt.imshow(recreate_image(centers, labels, rows, cols))


    #show original image
    plt.figure(0)
    plt.clf()
    plt.axis('off')
    plt.title('Original image')
    plt.imshow(image)

    # plot distance vs number of clusters
    plt.figure(11)
    plt.plot(range(1, 11), f_dist, 'go-', ms = 7)
    plt.title('Distance', fontsize = 40)
    plt.xlabel('Number of clusters', fontsize = 20)
    plt.ylabel('Intra cluster distance', fontsize = 20)
    plt.grid()
    plt.show()