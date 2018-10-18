
# start test like this:  nosetests-2.7  tests/test_TFUtil.py

from __future__ import print_function


import logging
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
import sys
sys.path += ["."]  # Python 3 hack
from TFUpdater import *
from TFNetworkLayer import LayerBase, Loss
from Log import log
from nose.tools import assert_equal, assert_is_instance, assert_is, assert_in
from numpy.testing.utils import assert_almost_equal
import unittest
import numpy.testing
import better_exchook

better_exchook.replace_traceback_format_tb()
log.initialize(verbosity=[5])

session = tf.InteractiveSession()


class DummyLoss(Loss):
  def get_value(self):
    assert self.layer and isinstance(self.layer, DummyLayer)
    return self.layer._get_loss_value()

  def get_error(self):
    return None


class DummyLayer(LayerBase):
  def __init__(self, initial_value=0.0, loss_value_factor=1.0, **kwargs):
    super(DummyLayer, self).__init__(**kwargs)
    self.loss_value_factor = loss_value_factor
    self.x = self.add_param(tf.Variable(initial_value))
    self.output.placeholder = self.x

  def _get_loss_value(self):
    return self.loss_value_factor * self.x

  @classmethod
  def get_losses(cls, name, network, output, loss=None, reduce_func=None, layer=None, **kwargs):
    assert not loss
    loss = DummyLoss(base_network=network)
    return super(DummyLayer, cls).get_losses(
      name=name, network=network, output=output, loss=loss, reduce_func=reduce_func, layer=layer, **kwargs)

  @classmethod
  def get_out_data_from_opts(cls, name, **kwargs):
    from TFUtil import Data
    return Data(name="%s_output" % name, batch_dim_axis=None, shape=(), dtype="float32")  # scalar


def test_Updater_GradientDescent():
  from TFNetwork import TFNetwork, ExternData
  from Config import Config

  config = Config()
  network = TFNetwork(extern_data=ExternData(), train_flag=True)
  network.add_layer(name="output", layer_class=DummyLayer, initial_value=5.0, loss_value_factor=3.0)
  network.initialize_params(session=session)

  updater = Updater(config=config, network=network)
  updater.set_learning_rate(1.0, session=session)
  updater.set_trainable_vars(network.get_trainable_params())
  updater.init_optimizer_vars(session=session)
  session.run(updater.get_optim_op())
  # One gradient descent step from 3.0 * x: gradient is 3, thus 5 - 3 = 2.
  assert_almost_equal(session.run(network.get_default_output_layer().output.placeholder), 2.0)


def test_Updater_CustomUpdate():
  from TFNetwork import TFNetwork, ExternData
  from Config import Config
  from TFUtil import CustomUpdate

  config = Config()
  network = TFNetwork(extern_data=ExternData(), train_flag=True)
  layer = network.add_layer(name="output", layer_class=DummyLayer, initial_value=4.0)
  assert isinstance(layer, DummyLayer)
  network.initialize_params(session=session)

  class CustomUpdateAdd13(CustomUpdate):
    def update_var(self, var):
      return tf.assign_add(var, 13.0)
  CustomUpdateAdd13().set_on_var(layer.x)

  updater = Updater(config=config, network=network)
  updater.set_learning_rate(1000.0, session=session)  # should be ignored
  updater.set_trainable_vars(network.get_trainable_params())
  updater.init_optimizer_vars(session=session)
  session.run(updater.get_optim_op())
  # Should have applied CustomUpdateAdd13.
  assert_almost_equal(session.run(network.get_default_output_layer().output.placeholder), 17.0)


def test_add_check_numerics_ops():
  with tf.Graph().as_default():
    with tf.Session().as_default() as session:
      x = tf.constant(3.0, name="x")
      y = tf.log(x * 3, name="y")
      assert isinstance(y, tf.Tensor)
      assert_almost_equal(session.run(y), numpy.log(9.))
      check = add_check_numerics_ops([y])
      session.run(check)
      z1 = tf.log(x - 3, name="z1")
      assert_equal(str(session.run(z1)), "-inf")
      z2 = tf.log(x - 4, name="z2")
      assert_equal(str(session.run(z2)), "nan")
      check1 = add_check_numerics_ops([z1])
      try:
        session.run(check1)
      except tf.errors.InvalidArgumentError as exc:
        print("Expected exception: %r" % exc)
      else:
        assert False, "should have raised an exception"
      check2 = add_check_numerics_ops([z2])
      try:
        session.run(check2)
      except tf.errors.InvalidArgumentError as exc:
        print("Expected exception: %r" % exc)
      else:
        assert False, "should have raised an exception"


def test_grad_add_check_numerics_ops():
  # Also see test_where_nan().
  with tf.Graph().as_default():
    with tf.Session().as_default() as session:
      x = tf.Variable(initial_value=0.0, name="x")
      session.run(x.initializer)
      y = 1. / x
      grad_x = tf.gradients(y, x)[0]
      print("grad_x:", grad_x.eval())
      assert_equal(str(float("-inf")), "-inf")
      assert_equal(str(grad_x.eval()), "-inf")

      session.run(x.assign(1.0))
      opt = tf.train.GradientDescentOptimizer(learning_rate=1.0)
      train_op = opt.minimize(y, var_list=[x])
      check = add_check_numerics_ops([train_op])
      session.run(check)

      session.run(x.assign(0.0))
      try:
        session.run(check)  # should fail now
      except tf.errors.InvalidArgumentError as exc:
        print("Expected exception: %r" % exc)
      else:
        assert False, "should have raised an exception"


def test_Updater_add_check_numerics_ops():
  class _Layer(DummyLayer):
    def _get_loss_value(self):
      return tf.log(self.x)

  from TFNetwork import TFNetwork, ExternData
  from Config import Config

  config = Config()
  config.set("debug_add_check_numerics_ops", True)
  network = TFNetwork(extern_data=ExternData(), train_flag=True)
  network.add_layer(name="output", layer_class=_Layer, initial_value=1.0)
  network.initialize_params(session=session)

  updater = Updater(config=config, network=network)
  updater.set_learning_rate(1.0, session=session)
  updater.set_trainable_vars(network.get_trainable_params())
  updater.init_optimizer_vars(session=session)
  # Should succeed.
  session.run(updater.get_optim_op())
  # One gradient descent step from ln(x), x = 1.0: gradient is 1.0 / x, thus x - 1.0 = 0.0.
  assert_almost_equal(session.run(network.get_default_output_layer().output.placeholder), 0.0)

  try:
    # Now, should fail.
    session.run(updater.get_optim_op())
  except tf.errors.InvalidArgumentError as exc:
    print("Expected exception: %r" % exc)
  else:
    assert False, "should have raised an exception"


if __name__ == "__main__":
  better_exchook.install()
  if len(sys.argv) <= 1:
    for k, v in sorted(globals().items()):
      if k.startswith("test_"):
        print("-" * 40)
        print("Executing: %s" % k)
        try:
          v()
        except unittest.SkipTest as exc:
          print("SkipTest:", exc)
        print("-" * 40)
    print("Finished all tests.")
  else:
    assert len(sys.argv) >= 2
    for arg in sys.argv[1:]:
      print("Executing: %s" % arg)
      if arg in globals():
        globals()[arg]()  # assume function and execute
      else:
        eval(arg)  # assume Python code and execute
