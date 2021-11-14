import numpy as np
import tensorflow as tf
# import tensorflow.contrib.eager as tfe

def data_splitter(points, labels, num_part1, num_classes=10):
    #Splitting data into two parts
    #For data with multiple classes, ensure that each part has equal number of examples of every class
    part1_data = {'input':[],'output':[]}
    part2_data = {'input':[],'output':[]}
    num_part2 = 0
    for class_ind in range(num_classes):
        #Labels are assumed not to be one-hot encoded
        class_points = points[np.argmax(labels,axis=1)==class_ind,:] #All images of class
        class_labels = labels[np.argmax(labels,axis=1)==class_ind,:] #All labels of class
        num_points = class_points.shape[0]
        num_part2 += num_points
        part1_pos = np.random.randint(num_points,size=int(num_part1/num_classes))
        part1_data['input'].append(class_points[part1_pos,:])
        part1_data['output'].append(class_labels[part1_pos,:])
        part2_pos = np.delete(np.arange(num_points),part1_pos)
        part2_data['input'].append(class_points[part2_pos,:])
        part2_data['output'].append(class_labels[part2_pos,:])
    num_part2 -= num_part1
    part1_data['input'] = np.concatenate(part1_data['input'])
    part1_data['output'] = np.concatenate(part1_data['output'])
    part2_data['input'] = np.concatenate(part2_data['input'])
    part2_data['output'] = np.concatenate(part2_data['output'])
    part1_ord = np.arange(num_part1)
    np.random.shuffle(part1_ord)
    part2_ord = np.arange(num_part2)
    np.random.shuffle(part2_ord)
    # print (lab_ord.shape)
    # print (lab_data['output'].shape)
    X1 = part1_data['input'][part1_ord,:]
    y1 = part1_data['output'][part1_ord,:]
    X2 = part2_data['input'][part2_ord,:]
    y2 = part2_data['output'][part2_ord,:]
    return X1, y1, X2, y2

def normalize(y):
    #Returns normalized data
    params = [np.mean(y,axis=0), np.std(y,axis=0)]
    return (y - params[0])/params[1], params

def build_input_pipeline(X_lab, y_lab, X_val, y_val, X_test, y_test, num_lab, num_val, num_test, batch_size):
  """Build an Iterator switching between train and heldout data."""

  # Build an iterator over training batches.
  training_dataset = tf.data.Dataset.from_tensor_slices((X_lab, np.int32(y_lab)))
  training_batches = training_dataset.shuffle(num_lab, reshuffle_each_iteration=True).repeat().batch(batch_size)
  training_iterator = tf.data.Dataset.make_one_shot_iterator(training_batches)

  # Build a iterator over the heldout set with batch_size=heldout_size,
  # i.e., return the entire heldout set as a constant.
  heldout_dataset = tf.data.Dataset.from_tensor_slices((X_val,np.int32(y_val)))
  heldout_frozen = (heldout_dataset.take(num_val).repeat().batch(num_val))
  heldout_iterator = tf.data.Dataset.make_one_shot_iterator(heldout_frozen)

  test_dataset = tf.data.Dataset.from_tensor_slices((X_test,np.int32(y_test)))
  test_frozen = (test_dataset.take(num_test).repeat().batch(num_test))
  test_iterator = tf.data.Dataset.make_one_shot_iterator(test_frozen)

  # Combine these into a feedable iterator that can switch between training
  # and validation inputs.
  handle = tf.placeholder(tf.string, shape=[])
  feedable_iterator = tf.data.Iterator.from_string_handle(handle, training_batches.output_types, training_batches.output_shapes)
  images, labels = feedable_iterator.get_next()
  return images, labels, handle, training_iterator, heldout_iterator, test_iterator