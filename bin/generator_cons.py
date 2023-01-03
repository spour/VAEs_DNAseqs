
def seq_temp_init(model, sequence_templates):
	"""
	Initializes sequence templates for a model.

	Copy code
	Parameters:
	model (Keras model): Model to initialize templates for.
	sequence_templates (list): List of sequence templates. Each template should be a string.

	Returns:
	None
	"""
	encoder = iso.OneHotEncoder(seq_length=len(templates[0]))
	embedding_templates = []
	embedding_masks = []
	for template in templates:
	    onehot_template = encoder(template).reshape((1, len(template), 4))
	    onehot_mask = np.zeros((1, len(template), 4))
	    for j, nt in enumerate(template):
	        if nt not in ['N', 'X']:
	            nt_ix = np.argmax(onehot_template[0, j, :])
	            onehot_template[:, j, :] = -4.0
	            onehot_template[:, j, nt_ix] = 10.0
	        elif nt == 'X':
	            onehot_template[:, j, :] = -1.0
	        elif nt == 'N':
	            onehot_mask[:, j, :] = 1.0
	    embedding_templates.append(onehot_template.reshape(1, -1))
	    embedding_masks.append(onehot_mask.reshape(1, -1))
	embedding_templates = np.concatenate(embedding_templates, axis=0)
	embedding_masks = np.concatenate(embedding_masks, axis=0)
	model.get_layer('template_dense').set_weights([embedding_templates])
	model.get_layer('template_dense').trainable = False
	model.get_layer('mask_dense').set_weights([embedding_masks])
	model.get_layer('mask_dense').trainable = False


def build_sampler(batch_size, seq_length, n_classes=1, n_samples=None, validation_sample_mode='max'):
    """
    Builds a Keras model for sampling DNA sequences from a position weight matrix (PWM).
    
    Parameters
    ----------
    batch_size: int
        The batch size to use when training the model.
    seq_length: int
        The length of the DNA sequences to generate.
    n_classes: int, optional
        The number of classes to classify DNA sequences into. The default is 1.
    n_samples: int, optional
        The number of samples to generate for each input DNA sequence. The default is None.
    validation_sample_mode: str, optional
        The sampling mode to use during validation. Can be either 'max' or 'sample'. The default is 'max'.
    
    Returns
    -------
    Keras model
        A Keras model that takes a PWM as input and generates one or more DNA sequences as output.
    """
    # Determine whether to generate one sample or multiple samples per input sequence
    use_samples = True
    if n_samples is None:
        use_samples = False
        n_samples = 1
    
    # Initialize a reshape layer to convert the input PWM into a 4D tensor
    reshape_layer = Reshape((1, seq_length, 4))
    
    # Initialize two embedding layers to create a template and a mask for the PWM
    onehot_template_dense = Embedding(n_classes, seq_length * 4, embeddings_initializer='zeros', name='dense_template')
    onehot_mask_dense = Embedding(n_classes, seq_length * 4, embeddings_initializer='ones', name='dense_mask')
    
    # Initialize a lambda layer to apply the template and mask to the PWM
    masking = Lambda(mask_pwm, output_shape=(1, seq_length, 4), name='masking')
    
    # Initialize a softmax layer to normalize the masked PWM
    pwm = Softmax(axis=-1, name='pwm')
    
    # Determine which sampling function to use
    if validation_sample_mode == 'sample':
        sample_func = only_sample_pwm
    else:
        sample_func = sample_pwm
    
    # Initialize a lambda layer to upsample the masked PWM
    upsampling = Lambda(lambda x: K.tile(x, [n_samples, 1, 1, 1]), name='upsampling')
    
    # Initialize a lambda layer to sample DNA sequences from the upsampled PWM
    sampling = Lambda(sample_func, name='pwm_sample')
    
    # Initialize a lambda layer to permute the dimensions of the sampled sequences
    permute = Lambda(lambda x: K.permute_dimensions(K.reshape(x, (n_samples, batch_size, 1, seq_length, 4)), (1, 0, 2, 3, 4)), name='permute')
    
    def _sample(input_tensor, raw_logits):
	    """
	    Samples from the predicted probability distribution over nucleotides at each position in the input tensor.

	    Parameters:
	    input_tensor (tensor): A tensor of shape (batch_size, sequence_length, alphabet_size)
	    raw_logits (tensor): A tensor of shape (batch_size, sequence_length, alphabet_size)
	    
	    Returns:
	    logits (tensor): A tensor of shape (batch_size, sequence_length, alphabet_size) representing the logits after being modified by the onehot template and mask.
	    pwm (tensor): A tensor of shape (batch_size, sequence_length, alphabet_size) representing the probability distribution over nucleotides at each position after being modified by the onehot template and mask.
	    sampled_pwm (tensor): A tensor of shape (batch_size, sequence_length, alphabet_size) representing the samples drawn from the pwm tensor. If use_samples=True, this tensor has an extra sample axis of size num_samples.
	    """
	    
	    # Get the onehot template and mask for this input
	    onehot_template = reshape_layer(onehot_template_dense(input_tensor))
	    onehot_mask = reshape_layer(onehot_mask_dense(input_tensor))
	    
	    # Add the template and multiply by the mask to modify the raw logits
	    logits = masking([raw_logits, onehot_template, onehot_mask])
	    
	    # Compute the probability distribution over nucleotides (nucleotide-wise softmax)
	    pwm = pwm(logits)
	    
	    # Optionally tile each PWM to sample from and create sample axis
	    if use_samples:
	        logits_upsampled = upsampling_layer(logits)
	        sampled_pwm = sampling_layer(logits_upsampled)
	        sampled_pwm = permute_layer(sampled_pwm)
	    else:
	        sampled_pwm = sampling_layer(logits)
	    
	    return logits, pwm, sampled_pwm
    
    return _sample


def pwm_cross_ent():
    """
    Computes the cross-entropy loss between two position weight matrices (PWMs).
    
    Parameters
    ----------
    pwm_true: tensor
        The true PWM.
    pwm_pred: tensor
        The predicted PWM.
    
    Returns
    -------
    tensor
        The cross-entropy loss between the true and predicted PWMs.
    """
    # Get the true and predicted PWMs from the input tensors
    true, pred = inputs
    
    # Clip the values in the predicted PWM to avoid division by zero
    pred = K.clip(pred, K.epsilon(), 1. - K.epsilon())
    
    # Calculate the cross-entropy loss between the true and predicted PWMs
    ce = - K.sum(true[:, 0, :, :] * K.log(pred[:, 0, :, :]), axis=-1)
    
    # Return the mean cross-entropy loss over all samples in the batch
    return K.expand_dims(K.mean(ce, axis=-1), axis=-1)


def weight_loss(loss_coeff=1.):
    """
    Scales a loss value by a given coefficient.
    
    Parameters
    ----------
    loss_coeff: float, optional
        The coefficient to scale the loss by. The default is 1.
    
    Returns
    -------
    float
        The scaled loss value.
    """
    # Return the loss value multiplied by the coefficient
    return loss_coeff * y_pred
    

def zsample(inputs):
    """
    Samples a latent vector from a Gaussian distribution with a given mean and log variance.
    
    Parameters
    ----------
    inputs: tuple
        A tuple containing the mean and log variance of the Gaussian distribution.
    
    Returns
    -------
    tensor
        A sample from the Gaussian distribution.
    """
    # Get the mean and log variance of the Gaussian distribution from the input tuple
    mean, log_var = inputs
    
    # Determine the batch size and latent dimension of the distribution
    batch_size = K.shape(mean)[0]
    latent_dim = K.int_shape(mean)[1]
    
    # Generate a random normal tensor with the same shape as the distribution
    epsilon = K.random_normal(shape=(batch_size, latent_dim))
    
    # Sample a latent vector from the distribution using the mean and log variance
    return mean + K.exp(0.5 * log_var) * epsilon


def zkl_loss(inputs):
    """
    Computes the KL divergence loss between two Gaussian distributions.
    
    Parameters
    ----------
    inputs: tuple
        A tuple containing the mean and log variance of the two Gaussian distributions.
    
    Returns
    -------
    tensor
        The KL divergence loss between the two Gaussian distributions.
    """
    # Get the mean and log variance of the two Gaussian distributions from the input tuple
    mean, log_var = inputs
    
    # Calculate the KL divergence loss
    kl_loss = 1 + log_var - K.square(mean) - K.exp(log_var)
    kl_loss = K.mean(kl_loss, axis=-1)
    kl_loss *= -0.5
    
    # Return the KL divergence loss as a tensor
    return K.expand_dims(kl_loss, axis=-1)
