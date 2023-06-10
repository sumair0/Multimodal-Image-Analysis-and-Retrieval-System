import numpy as np
import numpy
from numpy import linalg as LA
from collections import defaultdict
def lsh(vector_list, k, layers, vector_list_query, t, img_ids):
    #print(vector_list)
    print("len(img_ids):"+str(len(img_ids)))
    random_list = []
    layer_list = []
    layers_map = []
    num_dim = len(vector_list[0])
    layer_list=[]
    for run in range(layers):
        randv = numpy.random.randn(num_dim, k)
        random_list.append(randv)
        layer = np.matmul(vector_list, randv)
        layer = (layer > 0) * 1
        layer_list.append(layer)

    #hash map for each layer
    layers_map = []
    for lay in layer_list:
        layer_map = defaultdict(list)
        for row in range(len(lay)):
            st = str()
            for col in range(len(lay[0])):
                st += str(lay[row][col])
            layer_map[st].append((img_ids[row],vector_list[row]))
        layers_map.append(layer_map)
    ########
    #####5b########
    #converting query image to features: pass feature model here to query image and obtain features-192 for cm
    query_map = []
    query_list = []
    for layer in range(layers):
        randv = random_list[layer]
        query = np.matmul(vector_list_query, randv)#1*k matrix
        query = (query > 0) * 1
        query_list.append(query)

    # creating hash map for each layer
    for que in query_list:
        st = str()
        for i in range(len(que)):
            st += str(que[i])
        query_map.append(st)

    num_buckets = 0
    # search space for the query_image
    search_space = []
    for i, l in enumerate(layers_map):
        if query_map[i] in l:
            num_buckets+=1
            search_space.extend(l[query_map[i]])
    final_list = {}
    for li in search_space:
        final_list[li[0]] = li[1]
    # remove duplicates
    temp_final_list = []
    for name in final_list.keys():
        temp_final_list.append([name,final_list[name]])
    final_list = temp_final_list
    # print(temp_list)
    # final_list = set(final_list)

    # print(final_list)

    # similarity calculation
    results = []
    sim = []
    final_list_features = []
    # cosine similarity calculation
    for img_prop in (final_list):
        #get features of respective images 192*len(final_list)
        # final_list_features=get_feature_model(img_names)
        #get feautures of query image 192
        sim.append([np.dot(vector_list_query,img_prop[1])/(LA.norm(vector_list_query)*LA.norm(img_prop[1])),img_prop[0]])
    # top t similar images
    results = sorted(sim, key=lambda x: x[0], reverse=True)[:t]
    results = np.array(results)[:,1]
    expected_sim = []
    for i, feature in enumerate(vector_list):
        expected_sim.append([np.dot(vector_list_query,feature)/(LA.norm(vector_list_query)*LA.norm(feature)),img_ids[i]])

    exp_results = sorted(expected_sim, key=lambda x: x[0], reverse=True)[:t]
    exp_results = np.array(exp_results)[:,1]
    misses = 0
    for img_name in exp_results:
        if img_name not in results:
            misses+=1
    # misses = len(set(list(exp_results))-set(list(results)))
    total_unique_imaegs = len(final_list)
    print(results)
    #print(exp_results)
    print('Number of buckets',num_buckets)
    print('Miss_rate',misses*100/t,'%')
    print('False Positive', (total_unique_imaegs-(t-misses))*100/total_unique_imaegs, '%')
    # return np.array(results)[:,1]
