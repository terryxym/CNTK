from __future__ import print_function
import numpy as np
import cntk as C
from cntk import input_variable, parameter, Axis, NDArrayView
from cntk.learner import UserLearner, sgd, learning_rate_schedule, UnitType
from cntk.utils import ProgressPrinter
from cntk.layers import Dense, Sequential

        
SEED = 1

def generate_random_data(sample_size, feature_dim, num_classes):
    # Create synthetic data using NumPy.
    Y = np.random.randint(size=(sample_size, 1), low=0, high=num_classes)

    # Make sure that the data is separable
    X = (np.random.randn(sample_size, feature_dim) + 3) * (Y + 1)
    X = X.astype(np.float32)
    # converting class 0 into the vector "1 0 0",
    # class 1 into vector "0 1 0", ...
    class_ind = [Y == class_number for class_number in range(num_classes)]
    Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
    return X, Y


class MySgdNaive(UserLearner):

    def __init__(self, parameters, lr_schedule):
        super(MySgdNaive, self).__init__(parameters, lr_schedule)

    def update(self, gradient_values, training_sample_count, sweep_end):
        eta = self.learning_rate() / training_sample_count
        for p, g in gradient_values.items():
            new_p = p - eta * C.constant(g)
            p.set_value(new_p.eval(as_numpy=False).data())
        return True


class MySgdFast(UserLearner):

    def __init__(self, parameters, lr_schedule):
        super(MySgdFast, self).__init__(parameters, lr_schedule)

        self.new_p = {}
        self.param_input = {}
        self.grad_input = {}

        # we just need the batch axis
        ba = Axis.default_batch_axis()

        self.learning_rate_input = input_variable(1, dynamic_axes=[ba], name='lr')
        self.sample_count_input = input_variable(1, dynamic_axes=[ba], name='count')

        eta = self.learning_rate_input / self.sample_count_input

        # we need one graph per parameter shape
        for param in parameters:
            p_shape = param.shape
            self.param_input[p_shape] = input_variable(p_shape, dynamic_axes=[ba])
            self.grad_input[p_shape] = input_variable(p_shape, dynamic_axes=[ba])
            result = self.param_input[p_shape] - eta * self.grad_input[p_shape]
            self.new_p[p_shape] = result
            # from cntk.graph import plot
            # plot(self.new_p[p_shape], 'model_%s.pdf'%str(p_shape))

    def update(self, gradient_values, training_sample_count, sweep_end):
        for p, g in gradient_values.items():
            new_p = self.new_p[p.shape]
            param_input = self.param_input[p.shape]
            grad_input = self.grad_input[p.shape]

            data = {
                    param_input: np.asarray(p),
                    self.learning_rate_input: np.asarray([self.learning_rate()]),
                    self.sample_count_input: np.asarray([training_sample_count]),
                    grad_input: np.asarray(g)
                    }
            result = np.asarray(new_p.eval(data))
            p.set_value(NDArrayView.from_data(result[0][0]))

        return True


def ffnet(optimizer):
    inputs = 2
    outputs = 2
    hidden_dimension = 50

    # input variables denoting the features and label data
    features = C.input_variable((inputs), np.float32)
    label = C.input_variable((outputs), np.float32)

    # Instantiate the feedforward classification model
    my_model = Sequential([
        Dense(hidden_dimension, activation=C.sigmoid,
              init=C.glorot_uniform(seed=SEED)),
        Dense(outputs, init=C.glorot_uniform(seed=SEED))])
    z = my_model(features)

    ce = C.cross_entropy_with_softmax(z, label)
    pe = C.classification_error(z, label)

    # Instantiate the trainer object to drive the model training
    lr_per_minibatch = learning_rate_schedule(0.125, UnitType.minibatch)
    progress_printer = ProgressPrinter(0)
    trainer = C.Trainer(z, (ce, pe), [optimizer(
        z.parameters, lr_per_minibatch)], progress_printer)

    # Get minibatches of training data and perform model training
    minibatch_size = 25
    num_minibatches_to_train = 1000

    for i in range(num_minibatches_to_train):
        train_features, labels = generate_random_data(
            minibatch_size, inputs, outputs)
        # Specify the mapping of input variables in the model to actual
        # minibatch data to be trained with
        trainer.train_minibatch({features: train_features, label: labels})

    test_features, test_labels = generate_random_data(
        minibatch_size, inputs, outputs)
    avg_error = trainer.test_minibatch(
        {features: test_features, label: test_labels})
    print(' error rate on an unseen minibatch: {}'.format(avg_error))
    return z.parameters


def test_user_learner():
    np.random.seed(SEED)
    # sort based on shape (this works because all parameters have different
    # shapes)
    p1 = sorted([p.value for p in ffnet(sgd)], key=lambda x: x.shape)

    np.random.seed(SEED)
    p2 = sorted([p.value for p in ffnet(MySgdNaive)], key=lambda x: x.shape)

    np.random.seed(SEED)
    p3 = sorted([p.value for p in ffnet(MySgdFast)], key=lambda x: x.shape)

    for a, b, c in zip(p1, p2, p3):
        assert np.allclose(a, b)
        assert np.allclose(a, c)
