@tf.function
def textual_inversion(model, noise_scheduler, data):
    """Performs textual inversion using a given model and noise scheduler. Uses a gradient tape to calculate the mean squared error between predicted noise and actual noise,
     uses this loss to update the weights of the text encoder with the goal of only training the embedding of the placeholder token
    Arguments:
    - model: A model that takes in noisy latents, timestep embeddings, and the output of the text encoder, and predicts noise
    - noise_scheduler: A noise scheduler that adds noise to latents based on a given timestep
    - data: A tuple containing images and prompt embeddings
    Returns:
    - a dictionary containing the loss value of the model
    """


    images, prompt_embeddings = data

    with tf.GradientTape() as tape:
        
        ### creating embeddings out of the images 
        image_embeddings = training_image_encoder(images)
        ### pass the embeddings to the sampler and save some sammples in latents
        latents = sample_from_encoder_outputs(image_embeddings)
        ### match the latents with those used in the training of Stable Diffusion (just a random number they used in the training)
        latents = latents * 0.18215

        ### random noise in the same shape as latents
        noise = tf.random.normal(tf.shape(latents))
        
        ### get the batch dimension of our input data
        batch_dim = tf.shape(latents)[0]

        ### for each sample in the batch we choose a different random timestep to later determine the specific timestep embedding
        timesteps = tf.random.uniform((batch_dim,), minval=0, maxval=noise_scheduler.train_timesteps, dtype=tf.int64,)


        ### add the noise corresponding to the timestep to the latents by use of the scheduler
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        ### tensor containing all possible indices
        indices = get_position_ids()
        
        ### calculate the output of the encoder
        output_encoder = model.text_encoder([prompt_embeddings, indices])
        
        ### getting the timestep embeddings for each timestep
        timestep_embeddings = tf.map_fn(fn=get_timestep_embedding, elems=timesteps, fn_output_signature=tf.float32,)

        ### calculate the noise predictions with help of the latents, the time step embeddings and the output of the encoder
        noise_pred = model.diffusion_model([noisy_latents, timestep_embeddings, output_encoder])

        ### compute the mean squared error between the noise and the predicted noise and reduce it by taking the mean
        loss = tf.keras.losses.mean_squared_error(noise_pred, noise)
        loss = tf.reduce_mean(loss, axis=2)
        loss = tf.reduce_mean(loss, axis=1)
        loss = tf.reduce_mean(loss)

        ### load the the weights we want to train from the text encoder and calculate the gradients for them
        trainable_weights = model.text_encoder.trainable_weights
        gradients = tape.gradient(loss, trainable_weights)

        ### we only want to update the gradient of the placeholder token, therefore we create the tensor condition which has the value true for the index of the placeholder token (49408) and otherwise false
        condition = gradients[0].indices == 49408

        ### add an extra dimension to later zero out the gradients for other tokens
        condition = tf.expand_dims(condition, axis=-1)

        ### freeze the weights for all tokens by setting the gradients to 0 except for the placeholder token
        gradients[0] = tf.IndexedSlices(values=tf.where(condition, gradients[0].values,0), indices=gradients[0].indices, dense_shape=gradients[0].dense_shape)

        ### apply the gradients to the trainable weights of the encoder and thus only training the placeholder token's embedding
        optimizer.apply_gradients(zip(gradients, trainable_weights))

        return {"loss": loss}


### beta is the diffusion rate 
noise_scheduler = NoiseScheduler(
    ### beta_start determines the amount of noise added at the start of the denoising process
    beta_start=0.00085,
    ### beta_end at the end of the denoising process
    beta_end=0.012,
    ### the beta_schedule determines that the diffusion rate increases linearly
    beta_schedule="scaled_linear",
    train_timesteps=1000,
)
