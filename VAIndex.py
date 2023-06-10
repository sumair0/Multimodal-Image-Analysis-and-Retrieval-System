import numpy as np
import math
import sys

class VAIndex:
    def __init__(self, b, vectors, model):
        self.b = int(b)
        self.vectors = vectors
        self.model = model

        bjs = []
        d = self.vectors.shape[1]
        for j in range(1,d+1):
            bj = math.floor(self.b/d)
            if j <= self.b % d:
                bj += 1
            bjs.append(bj)
        self.bjs = np.asarray(bjs)

        # print(self.bjs.cumsum())

        # Create partition points
        self.partition_points = [] # Array of arrays
        for i in range(len(self.bjs)):
            num_partitions = (2**self.bjs[i])
            dim_data = self.vectors[:,i]

            quantiles = []
            for t in range(num_partitions + 1):
                quantiles.append(t / num_partitions)

            temp_points = []
            # Compute the bins for equally splitting the variance data
            for q in quantiles:
                temp_points.append(np.quantile(dim_data, q))

            self.partition_points.append(temp_points)
        # print(self.partition_points)

        self.vafile = ''
        # Assign bit representations
        for row in self.vectors:
            row_rep = ''
            for j in range(len(row)):
                index = np.searchsorted(self.partition_points[j], row[j]) - 1
                if index < 0:
                    index = 0
                # print('Index',index)
                # index = 0

                # print(row[j], self.partition_points[j][index])
                # print(j)
                # while row[j] > self.partition_points[j][index]:
                #     index += 1
                #     if index == len(self.partition_points[j]) - 1:
                #         break
                # print('Here')
                # if len(format(index, 'b').zfill(self.bjs[j])) > self.bjs[j]:
                #     print('Value',row[j])
                #     print(self.partition_points[j])
                #     print('format',format(index, 'b').zfill(self.bjs[j]))
                #     print('Target length', self.bjs[j])
                #     exit()
                # print('Size of vector: ', len(format(index, 'b').zfill(self.bjs[j]))) # Ensure proper length of bit representation
                row_rep += format(index, 'b').zfill(self.bjs[j]) # Ensure proper length of bit representation
                # print(row[j])
                # print(index)
            # print(len(row_rep))
            # exit()
            # Add data object to va-file index
            self.vafile += row_rep
        # Print size of index structure in bytes
        print('Index Structure Created')
        print('Total Size in Bytes: ' + str(len(self.vafile)))
        print()



    def initCandidate(self, t):
        for k in range(t):
            self.dst[k] = sys.maxsize
        return sys.maxsize

    def getBounds(self, approximation, query_feature, query_approximation):
        # Using Euclidean distance
        sum_term = 0
        bj_index = 0
        # print(approximation)
        for i in range(len(query_feature)):
            dim_bit_length = self.bjs[i]
            rij = int(approximation[bj_index:bj_index+dim_bit_length], 2)
            rqj = int(query_approximation[bj_index:bj_index+dim_bit_length], 2)
            bj_index += dim_bit_length
            if rij < rqj:
                sum_term += (query_feature[i] - self.partition_points[i][rij+1])**2
            if rij == rqj:
                sum_term += 0
            else:
                sum_term += (self.partition_points[i][rij+1] - query_feature[i])**2

        return sum_term ** (1/2)

    def candidate(self, n, norm, index):
        if norm < self.dst[n-1]:
            self.dst[n-1] = norm
            self.ans[n-1] = index
            # Sort based on distances
            zipped_lists = zip(self.dst, self.ans)
            sorted_pairs = sorted(zipped_lists)

            tuples = zip(*sorted_pairs)
            self.dst, self.ans = [list(tuple) for tuple in tuples]

            # print(dst)

        return self.dst[n-1]

    def query(self, query_feature, t):
        num_candidates_checked = 0
        # Get query regions
        query_rep = ''

        for j in range(len(query_feature)):
                index = np.searchsorted(self.partition_points[j], query_feature[j] ) - 1
                if index < 0:
                    index = 0
                query_rep += format(index, 'b').zfill(self.bjs[j])
        # for j in range(len(query_feature)):
        #     index = 0
        #     while query_feature[j] < self.partition_points[j][index]:
        #         index += 1


        self.dst = [0] * t
        self.ans = [0] * t
        d = self.initCandidate(t)
        for index in range(len(self.vectors)):
            current_approx = self.vafile[index*self.b:(index+1)*self.b]
            li = self.getBounds(current_approx, query_feature, query_rep)
            if li < d:
                # print(li, d)
                d = self.candidate(t, np.linalg.norm((query_feature-self.vectors[index]),ord=2), index)
                num_candidates_checked += 1
                # print(self.dst)

        return num_candidates_checked, self.ans
    def get_model(self):
        return self.model

