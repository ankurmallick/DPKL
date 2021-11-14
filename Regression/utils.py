import numpy as np

def data_splitter(points, labels, num_part1):
    #Splitting data into two parts
    num_part2 = 0
    num_points = points.shape[0]
    num_part2 += num_points
    part1_pos = np.random.choice(num_points,size=int(num_part1),replace=False)
    X1 = points[part1_pos,:]
    y1 = labels[part1_pos]
    part2_pos = np.delete(np.arange(num_points),part1_pos)
    X2 = points[part2_pos,:]
    y2 = labels[part2_pos]
    return X1, y1, X2, y2

def normalize(y):
    #Returns normalized data
    params = [np.reshape(np.mean(y),(1,-1)), np.reshape(np.std(y),(1,-1))]
    return (y - params[0])/params[1], params