import tensorflow as tf
from TFNetworkLayer import _ConcatInputLayer, get_concat_sources_data_template
from TFUtil import Data


class HMMFactorization(_ConcatInputLayer):

  layer_class = "hmm_factorization"

  def __init__(self, attention_weights, base_encoder_transformed, prev_state, prev_outputs, n_out, debug=False,
               threshold=None, transpose_and_average_att_weights=False, **kwargs):
    """
    HMM factorization as described in Parnia Bahar's paper.
    Out of rnn loop usage.
    Please refer to the demos to see the layer in use.
    :param LayerBase attention_weights: Attention weights of shape [I, J, B, 1]
    :param LayerBase base_encoder_transformed: Encoder, inner most dimension transformed to a constant size
    'intermediate_size'. Tensor of shape [J, B, intermediate_size]
    :param LayerBase prev_state: Previous state data, with the innermost dimension set to a constant size
    'intermediate_size'. Tensor of shape [I, B, intermediate_size].
    :param LayerBase prev_outputs: Previous output data with the innermost dimension set to a constant size
    'intermediate_size'. Tensor of shape [I, B, intermediate_size]
    :param bool debug: True/False, whether to print debug info or not
    :param float|None threshold: (float, >0), if not set to 'none', all attention values below this threshold will be
    set to 0. Slightly improves speed.
    :param bool transpose_and_average_att_weights: Set to True if using Transformer architecture. So, if
    attention_weights are of shape [J, B, H, I] with H being the amount of heads in the architecture. We will then
    average out over the heads to get the final attention values used.
    :param int n_out: Size of output dim (usually not set manually)
    :param kwargs:
    """

    super(HMMFactorization, self).__init__(**kwargs)

    # Get data
    self.attention_weights = attention_weights.output.get_placeholder_as_time_major()
    self.base_encoder_transformed = base_encoder_transformed.output.get_placeholder_as_time_major()
    self.prev_state = prev_state.output.get_placeholder_as_time_major()
    self.prev_outputs = prev_outputs.output.get_placeholder_as_time_major()

    if debug:
      self.attention_weights = tf.Print(self.attention_weights, [tf.shape(self.attention_weights)],
                                        message='Attention weight shape: ', summarize=100)

    if debug:
      self.base_encoder_transformed = tf.Print(self.base_encoder_transformed, [tf.shape(self.base_encoder_transformed),
                                                                               tf.shape(self.prev_state),
                                                                               tf.shape(self.prev_outputs)],
                                               message='Shapes of base encoder, prev_state and prev_outputs pre shaping: ',
                                               summarize=100)

    # Transpose and average out attention weights (for when we use transformer architecture)
    if transpose_and_average_att_weights is True:
      # attention_weights is [J, B, H, I]
      self.attention_weights = tf.transpose(self.attention_weights, perm=[3, 0, 1, 2])  # Now it is [I, J, B, H]
      self.attention_weights = tf.reduce_mean(self.attention_weights, keep_dims=True, axis=3)  # Now it is [I, J, B, 1]

      if debug:
        self.attention_weights = tf.Print(self.attention_weights, [tf.shape(self.attention_weights)],
                                          message='Attention weight shape after transposing and avg: ', summarize=100)

    # Get data
    attention_weights_shape = tf.shape(self.attention_weights)
    time_i = attention_weights_shape[0]
    batch_size = attention_weights_shape[2]
    time_j = attention_weights_shape[1]

    # Convert base_encoder_transformed, prev_state and prev_outputs to correct shape
    self.base_encoder_transformed = tf.tile(tf.expand_dims(self.base_encoder_transformed, axis=0),
                                            [time_i, 1, 1, 1])  # [I, J, B, intermediate_size]

    self.prev_state = tf.tile(tf.expand_dims(self.prev_state, axis=1),
                              [1, time_j, 1, 1])  # [I, J, B, intermediate_size]

    self.prev_outputs = tf.tile(tf.expand_dims(self.prev_outputs, axis=1),
                                [1, time_j, 1, 1])  # [I, J, B, intermediate_size]

    if debug:
      self.base_encoder_transformed = tf.Print(self.base_encoder_transformed, [tf.shape(self.base_encoder_transformed),
                                                                               tf.shape(self.prev_state),
                                                                               tf.shape(self.prev_outputs)],
                                               message='Shapes of base encoder, prev_state '
                                                       'and prev_outputs post shaping: ',
                                               summarize=100)

    # Permutate attention weights correctly
    self.attention_weights = tf.transpose(self.attention_weights, perm=[0, 2, 3, 1])  # Now [I, B, 1, J]

    if debug:
      self.attention_weights = tf.Print(self.attention_weights, [tf.shape(self.attention_weights)],
                                        message='attention_weights shape transposed: ',
                                        summarize=100)

    # Get logits, now [I, J, B, vocab_size]
    lexicon_logits = tf.layers.dense(self.base_encoder_transformed + self.prev_outputs + self.prev_state,
                                     units=n_out,
                                     activation=None,
                                     use_bias=False)
    lexicon_logits = tf.transpose(lexicon_logits, perm=[0, 2, 1, 3])  # Now [I, B, J, vocab_size]

    # Optimization with thresholding
    if threshold is not None:
      # Get mask
      amount_to_debug = 10

      if debug:
        self.attention_weights = tf.Print(self.attention_weights, [self.attention_weights],
                                          message='self.attention_weights: ', summarize=amount_to_debug)

      mask_values_to_keep = self.attention_weights > threshold  # Of shape [I, B, 1, J]

      # Modify mask to be of same shape as lexicon data
      mask_values_to_keep = tf.transpose(mask_values_to_keep, perm=[0, 1, 3, 2])  # Now [I, B, J, 1]
      mask_values_to_keep = tf.tile(mask_values_to_keep, [1, 1, 1, n_out])  # [I, B, J, vocab_size]

      if debug:
        mask_values_to_keep = tf.Print(mask_values_to_keep, [tf.shape(mask_values_to_keep)],
                                       message='mask_values_to_keep shape: ', summarize=100)
        mask_values_to_keep = tf.Print(mask_values_to_keep, [mask_values_to_keep],
                                       message='mask_values_to_keep: ', summarize=amount_to_debug)
        lexicon_logits = tf.Print(lexicon_logits, [lexicon_logits],
                                 message='lexicon_model pre mask: ', summarize=amount_to_debug)

      # Apply mask
      lexicon_logits = tf.where(mask_values_to_keep, lexicon_logits, tf.zeros_like(lexicon_logits))

      if debug:
        lexicon_logits = tf.Print(lexicon_logits, [lexicon_logits],
                                  message='lexicon_model post mask: ', summarize=amount_to_debug)

    lexicon_model = tf.nn.softmax(lexicon_logits)  # Now [I, B, J, vocab_size], Perform softmax on last layer

    if debug:
      lexicon_model = tf.Print(lexicon_model, [tf.shape(lexicon_model)], message='lexicon_model shape: ', summarize=100)

    # Multiply for final logits, [I, B, 1, J] x [I, B, J, vocab_size] ----> [I, B, 1, vocab]
    final_output = tf.matmul(self.attention_weights, lexicon_model)

    if debug:
      final_output = tf.Print(final_output, [tf.shape(final_output)], message='final_output shape: ', summarize=100)

    # Squeeze [I, B, vocab]
    final_output = tf.squeeze(final_output, axis=2)

    if debug:
      final_output = tf.Print(final_output, [tf.shape(final_output)], message='final_output post squeeze shape: ',
                              summarize=100)

    # Set shaping info
    if transpose_and_average_att_weights is True:
      output_size = self.input_data.size_placeholder[2]
      if debug:
        final_output = tf.Print(final_output, [self.input_data.size_placeholder[2]],
                                message='Prev output size placeholder: ',
                                summarize=100)
    else:
      output_size = self.input_data.size_placeholder[0]
      if debug:
        final_output = tf.Print(final_output, [self.input_data.size_placeholder[0]],
                                message='Prev output size placeholder: ',
                                summarize=100)

    self.output.placeholder = final_output

    self.output.size_placeholder = {
      0: output_size
    }
    self.output.time_dim_axis = 0
    self.output.batch_dim_axis = 1

    # Add all trainable params
    with self.var_creation_scope() as scope:
      self._add_all_trainable_params(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name))

  def _add_all_trainable_params(self, tf_vars):
    for var in tf_vars:
      self.add_param(param=var, trainable=True, saveable=True)

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    d.setdefault("from", [d["attention_weights"]])
    super(HMMFactorization, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["attention_weights"] = get_layer(d["attention_weights"])
    d["base_encoder_transformed"] = get_layer(d["base_encoder_transformed"])
    d["prev_state"] = get_layer(d["prev_state"])
    d["prev_outputs"] = get_layer(d["prev_outputs"])

  @classmethod
  def get_out_data_from_opts(cls, attention_weights, n_out, **kwargs):
    data = attention_weights.output
    data = data.copy_as_time_major()  # type: Data
    data.shape = (None, n_out)
    data.time_dim_axis = 0
    data.batch_dim_axis = 1
    data.dim = n_out
    return data
