'''network.py
Deep neural network core functionality implemented with the low-level TensorFlow API.
Saad Khan and Nithun Selva
CS444: Deep Learning
'''
import time
import numpy as np
import tensorflow as tf

from tf_util import arange_index

import sys
import platform

# Define a decorator function that applies different tf.function settings based on platform
# Cannot be bothered to keep changing this manually
def platform_aware_jit(func):
    # Check if running on macOS
    if platform.system() == 'Darwin':
        # macOS - no JIT
        # print('Running on macOS, no JIT compilation')
        return tf.function(func)
    else:
        # Other platforms - use JIT
        # print('Running using JIT compilation')
        return tf.function(jit_compile=True)(func)


class DeepNetwork:
    '''The DeepNetwork class is the parent class for specific networks (e.g. VGG).
    '''
    def __init__(self, input_feats_shape, reg=0):
        '''DeepNetwork constructor.

        Parameters:
        -----------
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            Example: If the input are 32x32 RGB images, input_feats_shape=(32, 32, 3).
        reg: float.
            The regularization strength.

        TODO: Set instance variables for the parameters passed into the constructor.
        '''
        # Keep these instance vars:
        self.optimizer_name = None
        self.loss_name = None
        self.output_layer = None
        self.all_net_params = []

        self.input_feats_shape = input_feats_shape
        self.reg = reg

    def compile(self, loss='cross_entropy', optimizer='adam', lr=1e-3, beta_1=0.9, print_summary=True):
        '''Compiles the neural network to prepare for training.

        This involves performing the following tasks:
        1. Storing instance vars for the loss function and optimizer that will be used when training.
        2. Initializing the optimizer.
        3. Doing a "pilot run" forward pass with a single fake data sample that has the same shape as those that will be
        used when training. This will trigger each weight layer's lazy initialization to initialize weights, biases, and
        any other parameters.
        4. (Optional) Print a summary of the network architecture (layers + shapes) now that we have initialized all the
        layer parameters and know what the shapes will be.
        5. Get references to all the trainable parameters (e.g. wts, biases) from all network layers. This list will be
        used during backpropogation to efficiently update all the network parameters.

        Parameters:
        -----------
        loss: str.
            Loss function to use during training.
        optimizer: str.
            Optimizer to use to train trainable parameters in the network. Initially supported options: 'adam'.
            NOTE: the 'adamw' option will be added later when instructed.
        lr: float.
            Learning rate used by the optimizer during training.
        beta_1: float.
            Hyperparameter in Adam and AdamW optimizers that controls the accumulation of gradients across successive
            parameter updates (in moving average).
        print_summary: bool.
            Whether to print a summary of the network architecture and shapes of activations in each layer.

        TODO: Fill in the section below that should create the supported optimizer. Use TensorFlow Keras optimizers.
        Assign the optimizer to the instance variable `opt`.
        '''
        self.loss_name = loss
        self.optimizer_name = optimizer

        # Initialize optimizer
        if optimizer.lower() == 'adam':
            self.opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1)
        elif optimizer.lower() == 'adamw':
            self.opt = tf.keras.optimizers.AdamW(learning_rate=lr, beta_1=beta_1, weight_decay=self.reg)
        else:
            raise ValueError(f'Unknown optimizer {optimizer}')

        # Do 'fake' forward pass through net to create wts/bias
        x_fake = self.get_one_fake_input()
        self(x_fake)

        # Initialize batch norm vars
        self.init_batchnorm_params()

        # Print network arch
        if print_summary:
            self.summary()

        # Get reference to all net params
        self.all_net_params = self.get_all_params()

    def get_one_fake_input(self):
        '''Generates a fake mini-batch of one sample to forward through the network when it is compiled to trigger
        lazy initialization to instantiate the weights and biases in each layer.

        This method is provided to you, so you should not need to modify it.
        '''
        return tf.zeros(shape=(1, *self.input_feats_shape))

    def summary(self):
        '''Traverses the network backward from output layer to print a summary of each layer's name and shape.

        This method is provided to you, so you should not need to modify it.
        '''
        print(75*'-')
        layer = self.output_layer
        while layer is not None:
            print(layer)
            layer = layer.get_prev_layer_or_block()
        print(75*'-')

    def set_layer_training_mode(self, is_training):
        '''Sets the training mode in each network layer.

        Parameters:
        -----------
        is_training: bool.
            True if the network is currently in training mode, False otherwise.

        TODO: Starting with the output layer, traverse the net backward, calling the appropriate method to
        set the training mode in each network layer. Model this process around the summary method.
        '''
        
        if is_training:
            # Set the mode in all layers to the training mode
            self.output_layer.set_mode(is_training)
            layer = self.output_layer.get_prev_layer_or_block()
            while layer is not None:
                layer.set_mode(is_training)
                layer = layer.get_prev_layer_or_block()
        else:
            # Set the mode in all layers to the non-training mode
            self.output_layer.set_mode(is_training)
            layer = self.output_layer.get_prev_layer_or_block()
            while layer is not None:
                layer.set_mode(is_training)
                layer = layer.get_prev_layer_or_block()

    def init_batchnorm_params(self):
        '''Initializes batch norm related parameters in all layers that are using batch normalization.

        (Week 3)

        TODO: Starting with the output layer, traverse the net backward, calling the appropriate method to
        initialize the batch norm parameters in each network layer. Model this process around the summary method.
        '''
        layer = self.output_layer
        while layer is not None:
            if hasattr(layer, 'init_batchnorm_params'):
                layer.init_batchnorm_params()
            layer = layer.get_prev_layer_or_block()
        pass

    def get_all_params(self, wts_only=False):
        '''Traverses the network backward from the output layer to compile a list of all trainable network paramters.

        This method is provided to you, so you should not need to modify it.

        Parameters:
        -----------
        wts_only: bool.
            Do we only collect a list of only weights (i.e. no biases or other parameters).

        Returns:
        --------
        Python list.
            List of all trainable parameters across all network layers.
        '''
        all_net_params = []

        layer = self.output_layer
        while layer is not None:
            if wts_only:
                params = layer.get_wts()

                if params is None:
                    params = []
                if not isinstance(params, list):
                    params = [params]
            else:
                params = layer.get_params()

            all_net_params.extend(params)
            layer = layer.get_prev_layer_or_block()
        return all_net_params

    def accuracy(self, y_true, y_pred):
        '''Computes the accuracy of classified samples. Proportion correct.

        Parameters:
        -----------
        y_true: tf.constant. shape=(B,).
            int-coded true classes.
        y_pred: tf.constant. shape=(B,).
            int-coded predicted classes by the network.

        Returns:
        -----------
        float.
            The accuracy in range [0, 1]

        Hint: tf.where might be helpful.
        '''
        
        acc = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))
        return acc

    def predict(self, x, output_layer_net_act=None):
        '''Predicts the class of each data sample in `x` using the passed in `output_layer_net_act`.
        If `output_layer_net_act` is not passed in, the method should compute it in order to perform the prediction.

        Parameters:
        -----------
        x: tf.constant. shape=(B, ...). Data samples
        output_layer_net_act: tf.constant. shape=(B, C) or None. Network activation.

        Returns:
        -----------
        tf.constant. tf.ints32. shape=(B,).
            int-coded predicted class for each sample in the mini-batch.
        '''
        if output_layer_net_act is None:
            output_layer_net_act = self(x)

        return tf.argmax(output_layer_net_act, axis=1)

    def loss(self, out_net_act, y, eps=1e-16):
        '''Computes the loss for the current minibatch based on the output layer activations `out_net_act` and int-coded
        class labels `y`.

        Parameters:
        -----------
        output_layer_net_act: tf.constant. shape=(B, C) or None.
            Net activation in the output layer for the current mini-batch.
        y: tf.constant. shape=(B,). tf.int32s.
            int-coded true classes for the current mini-batch.

        Returns:
        -----------
        float.
            The loss.

        TODO:
        1. Compute the loss that the user specified when calling compile. As of Project 1, the only option that
        should be supported/implemented is 'cross_entropy' for general cross-entropy loss.
        2. Throw an error if the the user specified loss is not supported.

        NOTE: I would like you to implement cross-entropy loss "from scratch" here — i.e. using the equation provided
        in the notebook, NOT using a TF high level function. For your convenience, I am providing the `arange_index`
        function in tf_util.py that offers functionality that is similar to arange indexing in NumPy (which you cannot
        do in TensorFlow). Use it!
        '''

        selected_values = arange_index(out_net_act, y)

        # Compute the cross-entropy loss
        loss = -tf.reduce_mean(tf.math.log(selected_values + eps))



        
        # else:
            # raise ValueError(f'Unknown loss function {self.loss_name}')

        # Keep the following code
        # Handles the regularization for Adam
        if self.optimizer_name.lower() == 'adam':
            all_net_wts = self.get_all_params(wts_only=True)
            reg_term = self.reg*0.5*tf.reduce_sum([tf.reduce_sum(wts**2) for wts in all_net_wts])
            loss = loss + reg_term
        return loss

    def update_params(self, tape, loss):
        '''Do backpropogation: have the optimizer update the network parameters recorded on `tape` based on the
        gradients computed of `loss` with respect to each of the parameters. The variable `self.all_net_params`
        represents a 1D list of references to ALL trainable parameters in every layer of the network
        (see compile method).

        Parameters:
        -----------
        tape: tf.GradientTape.
            Gradient tape object on which all the gradients have been recorded for the most recent forward pass.
        loss: tf.Variable. float.
            The loss computed over the current mini-batch.
        '''
        grads = tape.gradient(loss, self.all_net_params)
        self.opt.apply_gradients(zip(grads, self.all_net_params))
    
    # tf.function(jit_compile=True)
    # @tf.function
    @platform_aware_jit
    def train_step(self, x_batch, y_batch):
        '''Completely process a single mini-batch of data during training. This includes:
        1. Performing a forward pass of the data through the entire network.
        2. Computing the loss.
        3. Updating the network parameters using backprop (via update_params method).

        Parameters:
        -----------
        x_batch: tf.constant. tf.float32s. shape=(B, Iy, Ix, n_chans).
            A single mini-batch of data packaged up by the fit method.
        y_batch: tf.constant. tf.ints32. shape=(B,).
            int-coded labels of samples in the mini-batch.

        Returns:
        --------
        float.
            The loss.

        NOTE: Don't forget to record gradients on a gradient tape!
        '''
        with tf.GradientTape() as tape:
            out_net_act = self(x_batch)
            loss = self.loss(out_net_act, y_batch)
        self.update_params(tape, loss)        

        return loss
    
    # tf.function(jit_compile=True)
    # @tf.function
    @platform_aware_jit
    def test_step(self, x_batch, y_batch):
        '''Completely process a single mini-batch of data during test/validation time. This includes:
        1. Performing a forward pass of the data through the entire network.
        2. Computing the loss.
        3. Obtaining the predicted classes for the mini-batch samples.
        4. Compute the accuracy of the predictions.

        Parameters:
        -----------
        x_batch: tf.constant. tf.float32s. shape=(B, Iy, Ix, n_chans).
            A single mini-batch of data packaged up by the fit method.
        y_batch: tf.constant. tf.ints32. shape=(B,).
            int-coded labels of samples in the mini-batch.

        Returns:
        --------
        float.
            The accuracy.
        float.
            The loss.

        NOTE: There should not be any gradient tapes here.
        '''
        
        out_net_act = self(x_batch)
        loss = self.loss(out_net_act, y_batch)
        y_pred = self.predict(x_batch, out_net_act)
        y_pred = tf.cast(y_pred, tf.int32)
        y_batch = tf.cast(y_batch, tf.int32)
        acc = self.accuracy(y_batch, y_pred)

        return acc, loss
    
    def fit(self, x, y, x_val=None, y_val=None, batch_size=128, max_epochs=10000, val_every=1, verbose=True,
            patience=999, lr_patience=999, lr_decay_factor=0.5, lr_max_decays=12):
        '''Trains the neural network on the training samples `x` (and associated int-coded labels `y`).

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(N, Iy, Ix, n_chans).
            The data samples.
        y: tf.constant. tf.int32s. shape=(N,).
            int-coded class labels
        x_val: tf.constant. tf.float32. shape=(N_val, Iy, Ix, n_chans).
            Validation set samples.
        y_val: tf.constant. tf.float32. shape=(N_val,).
            int-coded validation set class labels.
        batch_size: int.
            Number of samples to include in each mini-batch.
        max_epochs: int.
            Network should train no more than this many epochs.
            Why it is not just called `epochs` will be revealed in Week 2.
        val_every: int.
            How often (in epoches) to compute validation set accuracy and loss.
        verbose: bool.
            If `False`, there should be no print outs during training. Messages indicating start and end of training are
            fine.
        patience: int.
            Number of most recent computations of the validation set loss to consider when deciding whether to stop
            training early (before `max_epochs` is reached).
            NOTE: Ignore Week 1 and until instructed otherwise.
        lr_patience: int.
            Number of most recent computations of the validation set loss to consider when deciding whether to decay the
            optimizer learning rate.
            NOTE: Ignore Week 1 and 2 and until instructed otherwise.
        lr_decay_factor: float.
            A value between 0.0. and 1.0 that represents the proportion of the current learning rate that the learning
            rate should be set to. For example, 0.7 would mean the learning rate is set to 70% its previous value.
            NOTE: Ignore Week 1 and 2 and until instructed otherwise.
        lr_max_decays: int.
            Number of times we allow the lr to decay during training.
            NOTE: Ignore Week 1 and 2 and until instructed otherwise.

        Returns:
        -----------
        train_loss_hist: Python list of floats. len=num_epochs.
            Training loss computed on each training mini-batch and averaged across all mini-batchs in one epoch.
        val_loss_hist: Python list of floats. len=num_epochs/val_freq.
            Loss computed on the validation set every time it is checked (`val_every`).
        val_acc_hist: Python list of floats. len=num_epochs/val_freq.
            Accuracy computed on the validation every time it is checked  (`val_every`).
        e: int.
            The number of training epochs used

        TODO:
        0. To properly handle Dropout layers in your network, set the mode of all layers in the network to train mode
        before the training loop begins.
        1. Process the data in mini-batches of size `batch_size` for each training epoch. Use the strategy recommended
        in CS343 for sampling the dataset randomly WITH replacement.
            NOTE: I suggest using NumPy to create a RNG (before the training loop) with a fixed random seed to
            generate mini-batch indices. That way you can ensure that differing results you get across training runs are
            not just due to your random choice of samples in mini-batches. This should probably be your ONLY use of
            NumPy in `DeepNetwork`.
        2. Call `train_step` to handle the forward and backward pass on the mini-batch.
        3. Average and record training loss values across all mini-batches in each epoch (i.e. one avg per epoch).
        4. If we are at the end of an appropriate epoch (determined by `val_every`):
            - Check and record the acc/loss on the validation set.
            - Print out: current epoch, training loss, val loss, val acc
        5. Regardless of `val_every`, print out the current epoch number (and the total number). Use the time module to
        also print out the time it took to complete the current epoch. Try to print the time and epoch number on the
        same line to reduce clutter.

        NOTE:
        - The provided `evaluate` method (below) should be useful for computing the validation acc+loss ;)
        - `evaluate` kicks all the network layers out of training mode (as is required bc it is doing prediction).
        Be sure to bring the network layers back into training mode after you are doing computing val acc+loss.
        '''
        # Set the mode in all layers to the training mode
        self.set_layer_training_mode(is_training=True)
    
        # Initialize training variables
        train_loss_hist, val_loss_hist, val_acc_hist = [], [], []
        recent_val_losses = [] # For early stopping
        recent_lr_val_losses = [] # For learning rate decay
        lr_decay_count = 0
        rng = np.random.default_rng(0)  # Fixed random seed for reproducibility
    
        # Determine the number of training samples
        N = x.shape[0]
        indices = np.arange(N)
    
        # Training loop
        for e in range(1, max_epochs + 1):  # Start from 1
            start_time = time.time()
            rng.shuffle(indices)
    
            # Mini-batch training
            epoch_loss = []
            for i in range(0, N, batch_size):
                batch_indices = indices[i:i+batch_size]
                x_batch = tf.gather(x, batch_indices)
                y_batch = tf.gather(y, batch_indices)
                
                loss = self.train_step(x_batch, y_batch)
                epoch_loss.append(loss.numpy())
            
            avg_train_loss = np.mean(epoch_loss)
            train_loss_hist.append(avg_train_loss)
    
            # Validation check
            if x_val is not None and y_val is not None and e % val_every == 0:
                val_acc, val_loss = self.evaluate(x_val, y_val)
                val_loss_hist.append(val_loss.numpy())
                val_acc_hist.append(val_acc.numpy())
                
                # Early stopping for training
                recent_val_losses, stop = self.early_stopping(recent_val_losses, val_loss, patience)
                if stop:
                    if verbose:
                        print(f'Early stopping at epoch {e}')
                    break
    
                
    
                # Print training progress
                if verbose:
                    print(f'Epoch {e}/{max_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
                self.set_layer_training_mode(is_training=True)

                # Early stopping for learning rate decay
                recent_lr_val_losses, lr_stop = self.early_stopping(recent_lr_val_losses, val_loss, lr_patience)
                if lr_stop and lr_decay_count < lr_max_decays:
                    self.update_lr(lr_decay_factor)
                    lr_decay_count += 1
            
            # Print epoch time
            end_time = time.time()
            if verbose:
                print(f'Epoch {e} completed in {end_time - start_time:.2f} seconds.')
        
        print(f'Finished training after {e} epochs!')
        return train_loss_hist, val_loss_hist, val_acc_hist, e
    
    def evaluate(self, x, y, batch_sz=64):
        '''Evaluates the accuracy and loss on the data `x` and labels `y`. Breaks the dataset into mini-batches for you
        for efficiency.

        This method is provided to you, so you should not need to modify it.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(N, Iy, Ix, n_chans).
            The complete dataset or one of its splits (train/val/test/dev).
        y: tf.constant. tf.ints32. shape=(N,).
            int-coded labels of samples in the complete dataset or one of its splits (train/val/test/dev).
        batch_sz: int.
            The batch size used to process the provided dataset. Larger numbers will generally execute faster, but
            all samples (and activations they create in the net) in the batch need to be maintained in memory at a time,
            which can result in crashes/strange behavior due to running out of memory.
            The default batch size should work fine throughout the semester and its unlikely you will need to change it.

        Returns:
        --------
        float.
            The accuracy.
        float.
            The loss.
        '''
        # Set the mode in all layers to the non-training mode
        self.set_layer_training_mode(is_training=False)

        # Make sure the mini-batch size isn't larger than the number of available samples
        N = len(x)
        if batch_sz > N:
            batch_sz = N

        num_batches = N // batch_sz

        # Make sure the mini-batch size is positive...
        if num_batches < 1:
            num_batches = 1

        # Process the dataset in mini-batches by the network, evaluating and avging the acc and loss across batches.
        loss = acc = 0
        for b in range(num_batches):
            curr_x = x[b*batch_sz:(b+1)*batch_sz]
            curr_y = y[b*batch_sz:(b+1)*batch_sz]

            curr_acc, curr_loss = self.test_step(curr_x, curr_y)
            acc += curr_acc
            loss += curr_loss
        acc /= num_batches
        loss /= num_batches

        return acc, loss

    def early_stopping(self, recent_val_losses, curr_val_loss, patience):
        '''Helper method used during training to determine whether training should stop before the maximum number of
        training epochs is reached based on the most recent loss values computed on the validation set
        (`recent_val_losses`) the validation loss on the current epoch (`curr_val_loss`) and `patience`.

        (Week 2)

        - When training begins, the recent history of validation loss values `recent_val_losses` is empty (i.e. `[]`).
        When we have fewer entries in `recent_val_losses` than the `patience`, then we just insert the current val loss.
        - The length of `recent_val_losses` should not exceed `patience` (only the most recent `patience` loss values
        are considered).
        - The recent history of validation loss values (`recent_val_losses`) is assumed to be a "rolling list" or queue.
        Remove the oldest loss value and insert the current validation loss into the list. You may keep track of the
        full history of validation loss values during training, but maintain a separate list in `fit()` for this.

        Conditions that determine whether to stop training early:
        - We never stop early when the number of validation loss values in the recent history list is less than patience
        (training is just starting out).
        - We stop early when the OLDEST rolling validation loss (`curr_val_loss`) is smaller than all recent validation
        loss values. IMPORTANT: Assume that `curr_val_loss` IS one of the recent loss values — so the oldest loss value
        should be compared with `patience`-1 other more recent loss values.

        Parameters:
        -----------
        recent_val_losses: Python list of floats. len between 0 and `patience` (inclusive).
            Recently computed losses on the validation set.
        curr_val_loss: float
            The loss computed on the validation set on the current training epoch.
        patience: int.
            The patience: how many recent loss values computed on the validation set we should consider when deciding
            whether to stop training early.

        Returns:
        -----------
        recent_val_losses: Python list of floats. len between 1 and `patience` (inclusive).
            The list of recent validation loss values passsed into this method updated to include the current validation
            loss.
        stop. bool.
            Should we stop training based on the recent validation loss values and the patience value?

        NOTE:
        - This method can be concisely implemented entirely with regular Python (TensorFlow/Numpy not needed).
        - It may be helpful to think of `recent_val_losses` as a queue: the current loss value always gets inserted
        either at the beginning or end. The oldest value is then always on the other end of the list.
        '''
        # Add current loss to history
        recent_val_losses.append(curr_val_loss)

        # Don't stop if we haven't collected enough losses yet
        if len(recent_val_losses) <= patience:
            return recent_val_losses, False

        # Keep only the most recent losses
        if len(recent_val_losses) > patience:
            recent_val_losses = recent_val_losses[-patience:]

        # Stop if the oldest loss is better than all subsequent losses
        old_val_loss = recent_val_losses[0]
        stop = all(loss > old_val_loss for loss in recent_val_losses[1:])

        return recent_val_losses, stop

    def update_lr(self, lr_decay_rate):
        '''Adjusts the learning rate used by the optimizer to be a proportion `lr_decay_rate` of the current learning
        rate.

        (Week 3)

        Paramters:
        ----------
        lr_decay_rate: float.
            A value between 0.0. and 1.0 that represents the proportion of the current learning rate that the learning
            rate should be set to. For example, 0.7 would mean the learning rate is set to 70% its previous value.

        NOTE: TensorFlow optimizer objects store the learning rate as a field called learning_rate.
        Example: self.opt.learning_rate for the optimizer object named self.opt. You are allowed to modify it with
        regular Python assignment.

        TODO:
        1. Update the optimizer's learning rate.
        2. Print out the optimizer's learning rate before and after the change.
        '''


        print('Current lr=', self.opt.learning_rate.numpy(), end=' ')
        self.opt.learning_rate.assign(self.opt.learning_rate * lr_decay_rate)
        print('Updated lr=', self.opt.learning_rate.numpy())

    def save_weights(self, file='weights.npy', verbose=False):
        """Save network weights to disk

        Parameters:
        -----------
        file: str
            File where weights should be saved
        verbose: bool
            If True, print out the names of the layers and parameters being saved
        """
        # Create dictionary to store all weights
        weights_dict = {}

        # Traverse the network to collect all parameters
        layer = self.output_layer
        layer_index = 0

        while layer is not None:
            # If the layer is a block, get the last layer in the block, and move up the block
            if hasattr(layer, 'blockname'):
                layer = layer.layers[-1]
            layer_name = layer.get_name()
            if verbose:
                print(f"Saving for {layer_name}")

            # Save weights if layer has them
            if hasattr(layer, 'wts') and layer.wts is not None:
                weights_dict[f"{layer_name}/weights"] = layer.wts.numpy()
                if verbose:
                    print(f"Saved weights for {layer_name}")

            # Save biases if layer has them
            if hasattr(layer, 'b') and layer.b is not None:
                weights_dict[f"{layer_name}/bias"] = layer.b.numpy()
                if verbose:
                    print(f"Saved bias for {layer_name}")

            # Save batch normalization parameters if they exist
            if hasattr(layer, 'bn_gain') and layer.bn_gain is not None:
                weights_dict[f"{layer_name}/bn_gain"] = layer.bn_gain.numpy()
                if verbose:
                    print(f"Saved bn_gain for {layer_name}")

            if hasattr(layer, 'bn_bias') and layer.bn_bias is not None:
                weights_dict[f"{layer_name}/bn_bias"] = layer.bn_bias.numpy()
                if verbose:
                    print(f"Saved bn_bias for {layer_name}")

            if hasattr(layer, 'bn_mean') and layer.bn_mean is not None:
                weights_dict[f"{layer_name}/bn_mean"] = layer.bn_mean.numpy()
                if verbose:
                    print(f"Saved bn_mean for {layer_name}")

            if hasattr(layer, 'bn_stdev') and layer.bn_stdev is not None:
                weights_dict[f"{layer_name}/bn_stdev"] = layer.bn_stdev.numpy()
                if verbose:
                    print(f"Saved bn_stdev for {layer_name}")
                    
            # Move to previous layer
            layer = layer.get_prev_layer_or_block()
            if verbose:
                print(f"Prev layer: {layer if layer else 'None'}")
            layer_index += 1

        # Save the weights dictionary
        np.save(f"{file}", weights_dict)
        print(f"Network weights saved to {file}")

    def load_weights(self, file='weights.npy', verbose=False):
        """Load network weights from disk

        Parameters:
        -----------
        file: str
            File from which weights should be loaded from
        verbose: bool
            If True, print out the names of the layers and parameters being loaded

        Returns:
        --------
        bool:
            True if weights were loaded successfully, False otherwise
        """
        # Check if file exists
        if not os.path.exists(f"{file}"):
            print(f"Error: Could not find weights file at {file}")
            return False

        # Load the weights dictionary
        try:
            weights_dict = np.load(f"{file}", allow_pickle=True).item()
        except:
            print(f"Error: Could not load weights from {file}")
            return False

        # Make sure we initialize network parameters before loading
        x_fake = self.get_one_fake_input()
        self(x_fake)

        # Traverse the network and load parameters
        layer = self.output_layer
        while layer is not None:
            if hasattr(layer, 'blockname'):
                # If the layer is a block, get the last layer in the block
                layer = layer.layers[-1]
            layer_name = layer.get_name()

            # Load weights if they exist
            if f"{layer_name}/weights" in weights_dict and hasattr(layer, 'wts'):
                layer.wts.assign(weights_dict[f"{layer_name}/weights"])
                if verbose:
                    print(f"Loaded weights for {layer_name}")

            # Load biases if they exist
            if f"{layer_name}/bias" in weights_dict and hasattr(layer, 'b'):
                layer.b.assign(weights_dict[f"{layer_name}/bias"])
                if verbose:
                    print(f"Loaded bias for {layer_name}")

            # Load batch normalization parameters if they exist
            if f"{layer_name}/bn_gain" in weights_dict and hasattr(layer, 'bn_gain') and layer.bn_gain is not None:
                layer.bn_gain.assign(weights_dict[f"{layer_name}/bn_gain"])
                if verbose:
                    print(f"Loaded bn_gain for {layer_name}")

            if f"{layer_name}/bn_bias" in weights_dict and hasattr(layer, 'bn_bias') and layer.bn_bias is not None:
                layer.bn_bias.assign(weights_dict[f"{layer_name}/bn_bias"])
                if verbose:
                    print(f"Loaded bn_bias for {layer_name}")

            if f"{layer_name}/bn_mean" in weights_dict and hasattr(layer, 'bn_mean') and layer.bn_mean is not None:
                layer.bn_mean.assign(weights_dict[f"{layer_name}/bn_mean"])
                if verbose:
                    print(f"Loaded bn_mean for {layer_name}")

            if f"{layer_name}/bn_stdev" in weights_dict and hasattr(layer, 'bn_stdev') and layer.bn_stdev is not None:
                layer.bn_stdev.assign(weights_dict[f"{layer_name}/bn_stdev"])
                if verbose:
                    print(f"Loaded bn_stdev for {layer_name}")

            # Move to previous layer
            layer = layer.get_prev_layer_or_block()

        print(f"Network weights loaded from {file}")
        return True