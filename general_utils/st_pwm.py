def stSampledSoftmax(logits):
	"""
	Samples a one-hot vector from the softmax distribution.

	Parameters:
	logits (tensor): Logits for the softmax distribution.

	Returns:
	tensor: Sampled one-hot vector.

	Explain: 
	It takes in a tensor logits, which are the logits for the softmax operation. It first applies the 
	softmax operation to logits to get the normalized probabilities nt_probs. It then samples a one-hot 
	tensor from these probabilities using the tf.random.categorical function. It then multiplies this 
	one-hot tensor with the normalized probabilities and applies the tf.math.ceil function to the 
	result. This is done within a gradient override context, so that the gradient of the ceil function 
	is overridden with the identity function during backpropagation. 
	"""
    nt_probs = tf.nn.softmax(logits)
	onehot_dim = logits.get_shape().as_list()[1]
	sampled_onehot = tf.one_hot(tf.squeeze(tf.random.categorical(logits, 1), 1), onehot_dim, 1.0, 0.0)
	with tf.compat.v1.get_default_graph().gradient_override_map({'Ceil': 'Identity', 'Mul': 'STMul'}):
	    return tf.math.ceil(sampled_onehot * nt_probs)

def sthardsoftmax(logits):
    """
    Computes a one-hot encoded tensor based on the hardmax of a softmax of the input tensor and multiplies it by the original softmax.
    The gradient of the `ceil` and `mul` operations is overriden.

    Parameters:
    logits (tensor): Input tensor.

    Returns:
    tensor: Result of the one-hot encoded tensor multiplied by the original softmax.

    """
    with ops.name_scope("StHardSoftmax") as namescope:
        # Compute softmax of input tensor
        softmax_logits = tf.nn.softmax(logits)

        # Compute one-hot encoded tensor based on hardmax of softmax
        onehot_dim = logits.get_shape().as_list()[1]
        onehot_encoded = tf.one_hot(tf.argmax(softmax_logits, 1), onehot_dim, 1.0, 0.0)

        # Multiply one-hot encoded tensor by original softmax and return result
        with tf.compat.v1.get_default_graph().gradient_override_map({'Ceil': 'Identity', 'Mul': 'STMul'}):
            return tf.math.ceil(onehot_encoded * softmax_logits)


@ops.RegisterGradient("STMul")
def stmul(op, grad):
	    """
    Defines the gradient function for the "STMul" operation.
    This function allows the gradient of the "STMul" operation to be passed through to its inputs.

    Parameters:
    op (TensorFlow Operation): TensorFlow operation for which the gradient is being computed.
    grad (Tensor): Tensor containing the gradient of the operation.

    Returns:
    Tensor: Tensor containing the gradient of the operation with respect to its inputs.
    """
    return [grad, grad]


#PWM Masking and Sampling helper functions

def mask_pwm(inputs) :
    pwm, onehot_template, onehot_mask = inputs
    return pwm * onehot_mask + onehot_template


def only_sample_pwm(logits) :
	"""
	Samples a PWM from the provided logits tensor.

	Copy code
	Parameters:
	logits (tensor): Tensor of logits for PWM sampling.

	Returns:
	tensor: Tensor of one-hot encoded PWM sequence.
	"""

    # Get number of sequences and length of each sequence
	n_sequences = K.shape(logits)[0]
	seq_length = K.shape(logits)[2]

	# Flatten PWM logits tensor
	flat_pwm = K.reshape(logits, (n_sequences * seq_length, 4))

	# Sample PWM from logits
	sampled_pwm = st_sampling_softmax(flat_pwm)

	# Reshape sampled PWM to original dimensions
	return K.reshape(sampled_pwm, (n_sequences, 1, seq_length, 4))

def sample_pwm(logits):
    """
    Samples a probability weight matrix (PWM) from logits.

    Parameters:
    logits (tensor): Logits for the PWM. Shape (batch size, 1, sequence length, 4)

    Returns:
    tensor: Sampled PWM. Shape (batch size, 1, sequence length, 4)
    """
    batch_size, _, seq_length, _ = K.shape(logits)

    # Flatten PWM logits
    flat_pwm = K.reshape(logits, (batch_size * seq_length, 4))
    
    # Sample PWM during training, use hardmax during testing
    sampled_pwm = K.switch(
        K.learning_phase(), 
        st_sampling_softmax(flat_pwm), 
        sthardsoftmax(flat_pwm)
    )

    return K.reshape(sampled_pwm, (batch_size, 1, seq_length, 4))
