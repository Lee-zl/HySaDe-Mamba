# hlsod数据集，先进行光谱聚类，再resize并保存

# from sklearn.mixture import GaussianMixture
# import glob
# from raw_norm import *
# from skimage import transform
# from skimage import measure
# import numpy as np
# import os
# import scipy.io as scio
# import h5py
# import hdf5plugin

# data_dir = '/home/zhangl/Datasets/HSODBIT-V1/hyperspectral/'
# dir_output = '/home/zhangl/PycharmProjects/DMSSN-main/DMSSN/SC_out/'
# name_list = glob.glob(data_dir+'*.h5')
# img_num = len(name_list)


# def normalize(data):
#     if data.shape[0] == 200:
#         for i in range(200):
#             data[i,:,:] = (data[i,:,:] - mean_raw[i]) / std_raw[i]
#     #data = data / np.max(data)
#     return data
# # def normalize(data):
# #     data_copy = np.zeros_like(data)
# #     if data_copy.shape[0] == 200:
# #         for i in range(200):
# #             data_copy[i,:,:] = (data[i,:,:] - mean_raw[i]) / std_raw[i]
# #     return data_copy


# def gaussian_mixture_model(inp_image, n_clusters=100, iters=10):
#     shape = inp_image.shape
#     inp_image = inp_image.flatten().reshape(shape[0]*shape[1], shape[2])
#     # Create Gaussian Mixture Model with Config Parameters
#     gmm = GaussianMixture(
#         n_components=n_clusters, covariance_type='full',
#         max_iter=iters, random_state=21)
#     # Fit on Input Image
#     gmm.fit(X=inp_image)
#     # Get Cluster Labels
#     clust = gmm.predict(X=inp_image)

#     return clust.reshape(shape[0], shape[1])


# for i in range(img_num):
#     name = name_list[i].split("/")[-1].split(".")[-2]
#     image = h5py.File(name_list[i], "r+")
#     # image = image["dataset"].value.astype('float32')
#     image2 = np.copy(image["dataset"])
#     image_norm = normalize(image2)
#     image_resize = transform.resize(image_norm, (200,512,512), preserve_range=True)

#     image_gmm = np.swapaxes(image_resize,0,2)
#     gmm = gaussian_mixture_model(image_gmm,n_clusters=50,iters=5)
#     label, sup = measure.label(gmm, background=0, return_num=True)

#     for cla in range(sup):
#         pos_list = np.where(label==sup)
#         x_list = pos_list[0]
#         y_list = pos_list[1]
#         val = np.zeros([1,1,image_gmm.shape[2]])
#         for num in range(len(x_list)):
#             x = x_list[num]
#             y = y_list[num]
#             val = val + image_gmm[x:x+1,y:y+1,:]
#         val = val / len(x_list)
#         for num in range(len(x_list)):
#             x = x_list[num]
#             y = y_list[num]
#             image_gmm[x:x+1,y:y+1,:] = val

#     image_result = np.swapaxes(image_gmm,0,2)
#     path = os.path.join(dir_output, name + ".mat")
#     scio.savemat(path, {'data':image_result})
#     print(i,name)




# import threading
# from sklearn.mixture import GaussianMixture
# import glob
# from raw_norm import *
# from skimage import transform
# from skimage import measure
# import numpy as np
# import os
# import scipy.io as scio
# import h5py
# import hdf5plugin

# os.environ["OPENBLAS_NUM_THREADS"] = "8"  # 设置线程数为64或更低

# data_dir = '/home/zhangl/Datasets/HSODBIT-V1/hyperspectral/'
# dir_output = '/home/zhangl/PycharmProjects/DMSSN-main/DMSSN/SC_out/'
# name_list = glob.glob(data_dir+'*.h5')
# img_num = len(name_list)


# def normalize(data):
#     if data.shape[0] == 200:
#         for i in range(200):
#             data[i,:,:] = (data[i,:,:] - mean_raw[i]) / std_raw[i]
#     #data = data / np.max(data)
#     return data


# def gaussian_mixture_model(inp_image, n_clusters=100, iters=10):
#     shape = inp_image.shape
#     inp_image = inp_image.flatten().reshape(shape[0]*shape[1], shape[2])
#     # Create Gaussian Mixture Model with Config Parameters
#     gmm = GaussianMixture(
#         n_components=n_clusters, covariance_type='full',
#         max_iter=iters, random_state=21)
#     # Fit on Input Image
#     gmm.fit(X=inp_image)
#     # Get Cluster Labels
#     clust = gmm.predict(X=inp_image)

#     return clust.reshape(shape[0], shape[1])


# def process_data(start_idx, end_idx):
#     for i in range(img_num):
#         name = name_list[i].split("/")[-1].split(".")[-2]
#         image = h5py.File(name_list[i], "r")
#         # image = image["dataset"].value.astype('float32')
#         image2 = np.array(image["dataset"].astype(np.float32))
#         image_norm = normalize(image2)
#         image_resize = transform.resize(image_norm, (200,512,512), preserve_range=True)

#         image_gmm = np.swapaxes(image_resize,0,2)
#         gmm = gaussian_mixture_model(image_gmm,n_clusters=50,iters=5)
#         label, sup = measure.label(gmm, background=0, return_num=True)

#         for cla in range(sup):
#             pos_list = np.where(label==sup)
#             x_list = pos_list[0]
#             y_list = pos_list[1]
#             val = np.zeros([1,1,image_gmm.shape[2]])
#             for num in range(len(x_list)):
#                 x = x_list[num]
#                 y = y_list[num]
#                 val = val + image_gmm[x:x+1,y:y+1,:]
#             val = val / len(x_list)
#             for num in range(len(x_list)):
#                 x = x_list[num]
#                 y = y_list[num]
#                 image_gmm[x:x+1,y:y+1,:] = val

#         image_result = np.swapaxes(image_gmm,0,2)
#         path = os.path.join(dir_output, name + ".mat")
#         scio.savemat(path, {'data':image_result})
#         print(i,name)

# # 定义线程数
# num_threads = 2
# # 计算每个线程需要处理的数据量
# chunk_size = img_num // num_threads

# # 创建线程
# threads = []
# for i in range(num_threads):
#     start_idx = i * chunk_size
#     # 最后一个线程处理剩余的数据
#     end_idx = start_idx + chunk_size if i < num_threads - 1 else img_num
#     t = threading.Thread(target=process_data, args=(start_idx, end_idx))
#     threads.append(t)

# # 启动线程
# for t in threads:
#     t.start()

# # 等待所有线程完成
# for t in threads:
#     t.join()

# print("All threads completed.")







import threading
from sklearn.mixture import GaussianMixture
import glob
from raw_norm import *
from skimage import transform
from skimage import measure
import numpy as np
import os
import scipy.io as scio
import h5py
import hdf5plugin

data_dir = '/home/zhangl/Datasets/HSODBIT-V1/hyperspectral/'
dir_output = '/home/zhangl/PycharmProjects/DMSSN-main/DMSSN/SC_out/'
name_list = glob.glob(data_dir+'*.h5')
img_num = len(name_list)


def normalize(data):
    if data.shape[0] == 200:
        for i in range(200):
            data[i,:,:] = (data[i,:,:] - mean_raw[i]) / std_raw[i]
    #data = data / np.max(data)
    return data


def gaussian_mixture_model(inp_image, n_clusters=100, iters=10):
    shape = inp_image.shape
    inp_image = inp_image.flatten().reshape(shape[0]*shape[1], shape[2])
    # Create Gaussian Mixture Model with Config Parameters
    gmm = GaussianMixture(
        n_components=n_clusters, covariance_type='full',
        max_iter=iters, random_state=21)
    # Fit on Input Image
    gmm.fit(X=inp_image)
    # Get Cluster Labels
    clust = gmm.predict(X=inp_image)

    return clust.reshape(shape[0], shape[1])


def process_data(start_idx, end_idx):
    for i in range(start_idx, end_idx):
        name = name_list[i].split("/")[-1].split(".")[-2]
        image = h5py.File(name_list[i], "r")
        image2 = np.array(image["dataset"].astype(np.float32))
        image_norm = normalize(image2)
        image_resize = transform.resize(image_norm, (200,512,512), preserve_range=True)

        image_gmm = np.swapaxes(image_resize,0,2)
        gmm = gaussian_mixture_model(image_gmm,n_clusters=50,iters=5)
        label, sup = measure.label(gmm, background=0, return_num=True)

        for cla in range(sup):
            pos_list = np.where(label==sup)
            x_list = pos_list[0]
            y_list = pos_list[1]
            val = np.zeros([1,1,image_gmm.shape[2]])
            for num in range(len(x_list)):
                x = x_list[num]
                y = y_list[num]
                val = val + image_gmm[x:x+1,y:y+1,:]
            val = val / len(x_list)
            for num in range(len(x_list)):
                x = x_list[num]
                y = y_list[num]
                image_gmm[x:x+1,y:y+1,:] = val

        image_result = np.swapaxes(image_gmm,0,2)
        path = os.path.join(dir_output, name + ".mat")
        scio.savemat(path, {'data':image_result})
        print(i,name)

# 定义线程数
num_threads = 4
# 计算每个线程需要处理的数据量
chunk_size = img_num // num_threads

# 创建线程
threads = []
for i in range(num_threads):
    start_idx = i * chunk_size
    # 最后一个线程处理剩余的数据
    end_idx = start_idx + chunk_size if i < num_threads - 1 else img_num
    t = threading.Thread(target=process_data, args=(start_idx, end_idx))
    threads.append(t)

# 启动线程
for t in threads:
    t.start()

# 等待所有线程完成
for t in threads:
    t.join()

print("All threads completed.")
