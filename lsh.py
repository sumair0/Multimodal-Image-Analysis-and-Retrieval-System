import numpy as np
import numpy
from numpy import linalg as LA

def lsh(vector_list, k, layers, vector_list_query, t, img_ids):
    random_list = []
    layer_list = []
    layers_map = []
    #TODO:get features of all 400 images as per feature model and replace that feature matrix will be the vector list here
    #vector_list=400*192 for cm
    #image-X-Y-Z.png
    #vector_list = []#have to append phase 1 results
    #img_ids=[]# append name of all imgs
    num_dim = len(vector_list[0])
    layer_list=[]
    for run in range(layers):
        randv = numpy.random.randn(num_dim, k)#192*k
        random_list.append(randv)#creating random list at every layer 192*k*l
        #each img converted into k bits
        layer = np.matmul(vector_list, randv)#400*k
        layer = (layer > 0) * 1#np arrays functionality - dont give anything it will be zero
        layer_list.append(layer)

    #hash map for each layer
    layers_map = []
    for lay in layer_list:
        layer_map = {}
        for row in range(len(lay)):
            st = str()
            for col in range(len(lay[0])):
                st += str(lay[row][col])
            layer_map[st] = img_ids[row]
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
        query_list.append(query)#we got for every layer 1 1*k matrix

    # creating hash map for each layer
    for que in query_list:
        st = str()
        for i in range(len(que)):
            st += str(que[i])
        query_map.append(st)

    # search space for the query_image
    search_space = []
    for i, l in enumerate(layers_map):
        search_space.append(l[query_map[i]])
    final_list = []
    for li in search_space:
        final_list += li
    # remove duplicates
    final_list = set(final_list)

    print(final_list)

    # similarity calculation
    results = []
    sim = []
    final_list_features = []
    # cosine similarity calculation
    '''
    @hanuman, I'm not sure what you are trying to perform cosine similarity for here,
    could you take a look at this and try to get it working?
    '''
    for img_names in (final_list):
        #get features of respective images 192*len(final_list)
        final_list_features=get_feature_model(img_names)
        #get feautures of query image 192
        sim.append(np.dot(vector_list_query,final_list_features)/(LA.norm(vector_list_query)*LA.norm(final_list_features)))
    # top t similar images
    results = sorted(sim, key=lambda x: x[1], reverse=True)[:t]
    return results
