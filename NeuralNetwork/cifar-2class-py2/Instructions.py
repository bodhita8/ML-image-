# Loading the data
# Now, the data should contain these items:
# dict["train_data"]: a 10000 x 3072 matrix with each row being a training image (you can visualize the image by reshaping the row to 32x32x3
# dict["train_labels"]: a 10000 x 1 vector with each row being the label of one training image, label 0 is an airplane, label 1 is a ship.
# dict["test_data"]: a 2000 x 3072 matrix with each row being a testing image
# dict["test_labels]: a 2000 x 1 vector with each row being the label of one testing image, corresponding to test_data.
import pickle as cPickle
import numpy as np

dict = cPickle.load(open("cifar_2class_py2.p","rb"))

for i in dict:
    print(i, dict[i].shape)

data = dict
train_x = data['train_data']
train_y = data['train_labels']
test_x = data['test_data']
test_y = data['test_labels']

sample = train_x[1,:]
print(sample.shape)

num_examples, input_dims = train_x.shape
print(num_examples)#10000
print(input_dims)#3072

# print(input_dims[0])
# print(input_dims[1])
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


if __name__ == "__main__":
    num_batches = 10
    count = 0
    for batch in iterate_minibatches(train_x, train_y, num_batches, shuffle=True):
        count += 1
        x_batch, y_batch = batch
        # print("\nBatch " + str(count))
        # print("x_batch shape")
        # print(x_batch.shape)
        print(y_batch.shape)
        print(count)
        W1 = np.random.uniform(low=-1, high=1, size=(input_dims + 1, 30))  # [3073, 31]
        b1 = np.random.randn(30)
# weights = np.random.random((1, 3072))
# print(weights.shape)
print(W1.shape)
print(b1.shape)