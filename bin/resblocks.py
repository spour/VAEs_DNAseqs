def gen_resblock(n_channels: int = 64, window_size: int = 3, 
                      stride: int = 1, dilation: int = 1, 
                      group_ix: int = 0, layer_ix: int = 0) -> callable:
    """
    Generates a residual block function with the given parameters.
    
    Parameters
    ----------
    n_channels: int
        The number of channels for the convolutional layers in the residual block.
    window_size: int
        The size of the convolutional kernel.
    stride: int
        The stride for the convolutional layers in the residual block.
    dilation: int
        The dilation rate for the convolutional layers in the residual block.
    group_ix: int
        The index of the group the residual block belongs to.
    layer_ix: int
        The index of the layer within the group.
    
    Returns
    -------
    callable
        The residual block function.
    """
    # Initialize res block layers
    batch_normal0 = tf.keras.layers.BatchNormalization(name=f'gen_resblock_{group_ix}_{layer_ix}_bn0')

    relu0 = tf.keras.layers.Lambda(lambda x: tf.keras.activations.relu(x))
    
    deconvolutional0 = tf.keras.layers.Conv2DTranspose(n_channels, (1, window_size), strides=(1, stride), 
                                                padding='same', activation='linear', 
                                                kernel_initializer='glorot_uniform', 
                                                name=f'gen_resblock_{group_ix}_{layer_ix}_deconv0')

    batch_normal1 = tf.keras.layers.BatchNormalization(name=f'gen_resblock_{group_ix}_{layer_ix}_bn1')

    relu1 = tf.keras.layers.Lambda(lambda x: tf.keras.activations.relu(x))

    convolutional1 = tf.keras.layers.Conv2D(n_channels, (1, window_size), dilation_rate=(1, dilation), strides=(1, 1), 
                                    padding='same', activation='linear', kernel_initializer='glorot_uniform', 
                                    name=f'gen_resblock_{group_ix}_{layer_ix}_conv1')

    skip_deconvolutional0 = tf.keras.layers.Conv2DTranspose(n_channels, (1, 1), strides=(1, stride), padding='same', activation='linear', 
    	kernel_initializer='glorot_uniform', name=f'gen_resblock_{group_ix}_{layer_ix}_skip_deconv0')

    skip1 = tf.keras.layers.Lambda(lambda x: x[0] + x[1], 
                                name=f'gen_resblock_{group_ix}_{layer_ix}_skip_')

	def _execute_residual_block(inputs):
	    """
	    Executes the residual block function on the given input tensor.
	    
	    Parameters
	    ----------
	    inputs: tensor
	        The input tensor to apply the residual block function to.
	        
	    Returns
	    -------
	    tensor
	        The output tensor after applying the residual block function.
	    """
	    norma_inputs = batch_normal0(inputs)
	    activ_inputs = relu_0(norm_inputs)
	    trans_conv_outputs = deconvolutional0(activ_inputs)

	    norm_trans_conv_outputs = batch_normal1(trans_conv_outputs)
	    activ_trans_conv_outputs = relu_1(norm_trans_conv_outputs)
	    conv_outputs = convolutional_1(activ_trans_conv_outputs)
	    
	    skip_outputs = skip_deconvolutional0(inputs)

	    added_outputs = skip1([conv_outputs, skip_outputs])
	    
	    return added_outputs

    return _execute_residual_block

#Decoder Model definition
def resnet_decoder(seq_length=256) :
    """
    Loads a decoder ResNet model.
    
    Parameters
    ----------
    seq_length: int
        The length of the sequence.
    latent_size: int
        The size of the latent space.
        
    Returns
    -------
    model
        The decoder ResNet model.
    """
    size_window = 3
    
    strides = [2, 2, 2, 2, 2, 1]
    dilations = [1, 1, 1, 1, 1, 1]
    channels = [384, seq_length, 128, 64, 32, 32]
    initial_length = 8
    nresblocks = len(strides)

    dense_0 = Dense(initial_length * channels[0], activation='linear', kernel_initializer='glorot_uniform', name='gen_dense0')
    dense_0_reshaped = Reshape((1, initial_length, channels[0]))
    
    curr_length = initial_length
    
    resblock = []
    for layer_ix in range(nresblocks):
        resblock.append(make_gen_resblock(n_channels=channels[layer_ix], size_window=size_window, stride=strides[layer_ix], dilation=dilations[layer_ix], group_ix=0, layer_ix=layer_ix))
    
    last_conv = Conv2D(4, (1, 1), strides=(1, 1), padding='same', activation='linear', kernel_initializer='glorot_uniform', name='gen_last_conv')    
		
		def generate(seed_input):
		  """
		  Generates an output tensor using the given seed input tensor.
		  
		  Parameters
		  ----------
		  seed_input: tensor
		      The seed input tensor to use for generating the output tensor.
		      
		  Returns
		  -------
		  tensor
		      The generated output tensor.
		  """
		  reshaped_output = dense_0_reshaped(dense_0(seed_input))
		      
		  # Connect group of res blocks
		  output_tensor = reshaped_output

		  # Res block group 0
		  for layer_ix in range(nresblocks):
		      output_tensor = resblocks[layer_ix](output_tensor)

		  # Final conv out
		  last_conv_out = last_conv(output_tensor) # last_conv(final_relu_out)
		      
		  return last_conv_out

    return _generate



def disc_res_block(channels: int, window: int, dilation: int, group: int, layer: int) -> None:
    """
    Initialize res block layers for a discriminator model.

    Parameters:
    channels (int): The number of channels in the convolutional layers.
    window (int): The window size for the convolutional layers.
    dilation (int): The dilation rate for the convolutional layers.
    group (int): The group index for the res block.
    layer (int): The layer index for the res block.
    """
    # Initialize batch normalization layer for res block
    bn0 = BatchNormalization(name=f'disc_resblock_{group}_{layer}_bn0')

    # Initialize ReLU activation function
    relu0 = Lambda(lambda x: K.relu(x, alpha=0.0))

    # Initialize first convolutional layer for res block
    conv0 = Conv2D(channels, (1, window), dilation_rate=dilation, strides=(1, 1), padding='same', 
                    activation='linear', kernel_initializer='glorot_normal', name=f'disc_resblock_{group}_{layer}_conv0')

    # Initialize second batch normalization layer for res block
    batch_norm_1 = BatchNormalization(name=f'disc_resblock_{group}_{layer}_bn1')

    # Initialize second ReLU activation function
    relu1 = Lambda(lambda x: K.relu(x, alpha=0.0))

    # Initialize second convolutional layer for res block
    conv1 = Conv2D(channels, (1, window), dilation_rate=dilation, strides=(1, 1), padding='same', 
                    activation='linear', kernel_initializer='glorot_normal', name=f'disc_resblock_{group}_{layer}_conv1')

    # Initialize skip connection for res block
    skip1 = Lambda(lambda x: x[0] + x[1], name=f'disc_resblock_{group}_{layer}_skip1')

    def _resblock(input_tensor):
		  """Residual block with two convolutional layers and skip connection.

		  Parameters
		  ----------
		  input_tensor : tensor
		      Input tensor to the residual block.

		  Returns
		  -------
		  tensor
		      Output tensor from the residual block.
		  """
		  # Initialize layers
		  bn0 = BatchNormalization()
		  relu0 = Lambda(lambda x: K.relu(x))
		  conv0 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='linear')
		  
		  batch_norm_1 = BatchNormalization()
		  relu1 = Lambda(lambda x: K.relu(x))
		  conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='linear')
		  
		  skip_1 = Lambda(lambda x: x[0] + x[1])

		  # Apply layers
		  bn0_out = bn0(input_tensor)
		  relu0_out = relu0(bn0_out)
		  conv0_out = conv0(relu0_out)

		  bn1_out = batch_norm_1(conv0_out)
		  relu1_out = relu1(bn1_out)
		  conv1_out = conv1(relu1_out)

		  skip1_out = skip_1([conv1_out, input_tensor])
		  return skip1_out

    return _resblock


def enc_net_4resblock(batch_size, seq_length=205, latent_size=100, drop_rate=0.25):
    """
    Loads a discriminator network with 4 resblock for encoding sequences.

    Parameters:
    batch_size (int): Batch size for the network.
    seq_length (int): Length of the input sequences. Default is 205.
    latent_size (int): Size of the latent space representation. Default is 100.
    drop_rate (float): Dropout rate. Default is 0.25.

    Returns:
    model: Discriminator network model.
    """

    # Discriminator network parameters

    n_resblock = 4
    nchannels = 32

    # Initialize first convolutional layer
    policy_conv0 = Conv2D(nchannels, (1, 1), strides=(1, 1), padding='same', activation='linear', kernel_initializer='glorot_normal', name='policy_discriminator_conv0')
    
    # Initialize skip connection convolutional layer
    skip_conv0 = Conv2D(nchannels, (1, 1), strides=(1, 1), padding='same', activation='linear', kernel_initializer='glorot_normal', name='policy_discriminator_skip_conv0')
    
    # Initialize res block layers
    resblock = []
    for layer_index in range(n_resblock):
        resblock.append(disc_res_block(nchannels=nchannels, size_window=8, dilation_rate=1, group_ix=0, layer_ix=layer_index))
    
    # Initialize last convolutional layer
    final_block_conv = Conv2D(nchannels, (1, 1), strides=(1, 1), padding='same', activation='linear', kernel_initializer='glorot_normal', name='policy_discriminator_final_block_conv')
    
    # Initialize skip connection
    add_skip = Lambda(lambda x: x[0] + x[1], name='policy_discriminator_add_skip')
    
    # Flatten output
    last_flatten = Flatten()
    
    # Initialize dense layers for mean and log variance of latent space representation
    z_mean = Dense(latent_size, name='policy_discriminator_z_mean')
    z_log_var = Dense(latent_size, name='policy_discriminator_z_log_var')
    
    	def _encoder(input_sequence):
		  """
		  Encodes an input sequence into latent space representation.

		  Parameters:
		  input_sequence (tensor): Input sequence tensor.

		  Returns:
		  tuple: Tuple containing tensors for the mean and log variance of the latent space representation.
		  """

		  # Apply first convolutional layer
		  conv0_output = policy_conv0(input_sequence)

		  # Connect group of res blocks
		  output_tensor = conv0_output

		  # Res block group 0
		  skip_conv0_output = skip_conv0(output_tensor)

		  # Apply res blocks
		  for layer_index in range(n_resblocks):
		      output_tensor = resblocks[layer_index](output_tensor)

		  # Last res block extr conv
		  final_block_conv_output = final_block_conv(output_tensor)

		  # Skip connection
		  skip_add_output = skip_add([final_block_conv_output, skip_conv0_output])

		  # Flatten output
		  last_dense_output = last_flatten(skip_add_output)

		  # Z mean and log variance
		  output_zmean = z_mean(last_dense_output)
		  output_zlog = z_log_var(last_dense_output)

		  return output_zmean, output_zlog

    return _encoder
