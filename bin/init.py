def initialize_models(batch_size, seq_length, latent_size, n_samples, sequence_templates):
	#Load Encoder
	encoder = enc_net_4resblocks(batch_size, seq_length=seq_length, latent_size=latent_size, drop_rate=0.)

	#Load Decoder
	decoder = resnet_decoder(seq_length=seq_length)

	#Load Sampler
	sampler = build_sampler(batch_size, seq_length, n_classes=1, n_samples=n_samples, validation_sample_mode='sample')

	#Build Encoder Model
	encoder_input = Input(shape=(1, seq_length, 4), name='encoder_input')

	z_mean, z_log_var = encoder(encoder_input)

	z_sampling_layer = Lambda(zsample, output_shape=(latent_size,), name='z_sampler')
	z = z_sampling_layer([z_mean, z_log_var])

	instantiate encoder model
	encoder_model = Model(encoder_input, [z_mean, z_log_var, z])
	encoder_model.compile(
	optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
	loss=min_pred
	)

	#Build Decoder Model
	decoder_class = Input(shape=(1,), name='decoder_class')
	decoder_input = Input(shape=(latent_size,), name='decoder_input')

	logits, pwm, sampled_pwm = sampler(decoder_class, decoder(decoder_input))

	decoder_model = Model([decoder_class, decoder_input], [logits, pwm, sampled_pwm])

	#Initialize Sequence Templates and Masks
	seq_temp_init(decoder_model, sequence_templates)

	decoder_model.compile(
	optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
	loss=min_pred
	)

	#Build VAE Pipeline

	vae_decoder_class = Input(shape=(1,), name='vae_decoder_class')
	vae_encoder_input = Input(shape=(1, seq_length, 4), name='vae_encoder_input')

	encoded_z_mean, encoded_z_log_var = encoder(vae_encoder_input)
	encoded_z = z_sampling_layer([encoded_z_mean, encoded_z_log_var])
	decoded_logits, decoded_pwm, decoded_sample = sampler(vae_decoder_class, decoder(encoded_z))

	reconstruction_loss = Lambda(pwm_cross_ent(), name='reconstruction')([vae_encoder_input, decoded_pwm])

	kl_loss = Lambda(zkl_loss(), name='kl')([encoded_z_mean, encoded_z_log_var])

	vae_model = Model(
	[vae_decoder_class, vae_encoder_input],
	[reconstruction_loss, kl_loss]#, entropy_loss]
	)

	#Initialize Sequence Templates and Masks
	seq_temp_init(vae_model, sequence_templates)

	vae_model.compile(
	optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.5, beta_2=0.9),
	loss={
	'reconstruction' : weight_loss(loss_coeff=1. * (1./147.)),
	'kl' : weight_loss(loss_coeff=0.65 * (1./147.))
	}
	)

	encoder_model.summary()

	decoder_model.summary()

	return encoder_model, decoder_model, vae_model
