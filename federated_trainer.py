# Copyright 2021, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runs federated training on ?? dataset."""

import functools
from typing import Any, Callable, Optional

from absl import app, flags, logging

import tensorflow as tf
import tensorflow_federated as tff

import training_loop
from federated_model import create_resnet50
from federated_dataset import get_federated_datasets, get_centralized_datasets

from utils import utils_impl
from utils.optimizers import optimizer_utils



with utils_impl.record_hparam_flags() as optimizer_flags:
  # Defining optimizer flags
  optimizer_utils.define_optimizer_flags('client')
  optimizer_utils.define_optimizer_flags('server')
  optimizer_utils.define_lr_schedule_flags('client')
  optimizer_utils.define_lr_schedule_flags('server')

with utils_impl.record_hparam_flags() as shared_flags:
  # Federated training hyperparameters
  flags.DEFINE_integer('num_classes', 100,
                       'Number of classes.')
  flags.DEFINE_integer('client_epochs_per_round', 1,
                       'Number of epochs in the client to take per round.')
  flags.DEFINE_integer('client_batch_size', 20, 'Batch size on the clients.')
  flags.DEFINE_integer('clients_per_round', 10,
                       'How many clients to sample per round.')
  flags.DEFINE_integer('client_datasets_random_seed', 1,
                       'Random seed for client sampling.')

  # Training loop configuration
  flags.DEFINE_string(
      'experiment_name', None, 'The name of this experiment. Will be append to '
      '--root_output_dir to separate experiment results.')
  flags.mark_flag_as_required('experiment_name')
  flags.DEFINE_string('root_output_dir', '/tmp/fed_opt/',
                      'Root directory for writing experiment output.')
  flags.DEFINE_integer('total_rounds', 200, 'Number of total training rounds.')
  flags.DEFINE_integer(
      'rounds_per_eval', 1,
      'How often to evaluate the global model on the validation dataset.')
  flags.DEFINE_integer('rounds_per_checkpoint', 50,
                       'How often to checkpoint the global model.')

with utils_impl.record_hparam_flags() as cifar10_flags:
  # CIFAR-10 flags
  flags.DEFINE_integer('crop_size', 24, 'The height and width of '
                       'images after preprocessing.')
  flags.DEFINE_boolean(
      'uniform_weighting', False,
      'Whether to weigh clients uniformly. If false, clients '
      'are weighted by the number of samples.')

FLAGS = flags.FLAGS


def _get_hparam_flags():
  """Returns an ordered dictionary of pertinent hyperparameter flags."""
  hparam_dict = utils_impl.lookup_flag_values(shared_flags)

  # Update with optimizer flags corresponding to the chosen optimizers.
  opt_flag_dict = utils_impl.lookup_flag_values(optimizer_flags)
  opt_flag_dict = optimizer_utils.remove_unused_flags('client', opt_flag_dict)
  opt_flag_dict = optimizer_utils.remove_unused_flags('server', opt_flag_dict)
  hparam_dict.update(opt_flag_dict)

  # Update with task-specific flags.
  task_hparam_dict = utils_impl.lookup_flag_values(cifar10_flags)
  hparam_dict.update(task_hparam_dict)

  return hparam_dict

def run_federated(
    iterative_process_builder: Callable[..., tff.templates.IterativeProcess],
    num_classes: int,
    client_epochs_per_round: int,
    client_batch_size: int,
    clients_per_round: int,
    client_datasets_random_seed: Optional[int] = None,
    crop_size: Optional[int] = 24,
    total_rounds: Optional[int] = 1500,
    experiment_name: Optional[str] = 'federated_experiment',
    root_output_dir: Optional[str] = '/tmp/fed_opt',
    uniform_weighting: Optional[bool] = False,
    **kwargs):
  """Runs an iterative process on the CIFAR-10 classification task.

  This method will load and pre-process dataset and construct a model used for
  the task. It then uses `iterative_process_builder` to create an iterative
  process that it applies to the task, using
  `federated_research.utils.training_loop`.
  We assume that the iterative process has the following functional type
  signatures:
    *   `initialize`: `( -> S@SERVER)` where `S` represents the server state.
    *   `next`: `<S@SERVER, {B*}@CLIENTS> -> <S@SERVER, T@SERVER>` where `S`
        represents the server state, `{B*}` represents the client datasets,
        and `T` represents a python `Mapping` object.
  The iterative process must also have a callable attribute `get_model_weights`
  that takes as input the state of the iterative process, and returns a
  `tff.learning.ModelWeights` object.

  Args:
    iterative_process_builder: A function that accepts a no-arg `model_fn`, and
      returns a `tff.templates.IterativeProcess`. The `model_fn` must return a
      `tff.learning.Model`.
    client_epochs_per_round: An integer representing the number of epochs of
      training performed per client in each training round.
    client_batch_size: An integer representing the batch size used on clients.
    clients_per_round: An integer representing the number of clients
      participating in each round.
    client_datasets_random_seed: An optional int used to seed which clients are
      sampled at each round. If `None`, no seed is used.
    crop_size: An optional integer representing the resulting size of input
      images after preprocessing.
    total_rounds: The number of federated training rounds.
    experiment_name: The name of the experiment being run. This will be appended
      to the `root_output_dir` for purposes of writing outputs.
    root_output_dir: The name of the root output directory for writing
      experiment outputs.
    uniform_weighting: Whether to weigh clients uniformly. If false, clients are
      weighted by the number of samples.
    **kwargs: Additional arguments configuring the training loop. For details on
      supported arguments, see `federated_research/utils/training_utils.py`.
  """

  crop_shape = (crop_size, crop_size, 3)

  cifar_train, _ = get_federated_datasets(
      train_client_epochs_per_round=client_epochs_per_round,
      train_client_batch_size=client_batch_size,
      crop_shape=crop_shape)

  _, cifar_test = get_centralized_datasets(
      crop_shape=crop_shape)

  input_spec = cifar_train.create_tf_dataset_for_client(
      cifar_train.client_ids[0]).element_spec

  model_builder = functools.partial(
      create_resnet50,
      input_shape=crop_shape,
      num_classes=num_classes)

  loss_builder = tf.keras.losses.SparseCategoricalCrossentropy
  metrics_builder = lambda: [tf.keras.metrics.SparseCategoricalAccuracy()]

  def tff_model_fn() -> tff.learning.Model:
    return tff.learning.from_keras_model(
        keras_model=model_builder(),
        input_spec=input_spec,
        loss=loss_builder(),
        metrics=metrics_builder())

  if uniform_weighting:
    client_weight_fn = tff.learning.ClientWeighting.UNIFORM
  else:
    client_weight_fn = tff.learning.ClientWeighting.NUM_EXAMPLES

  training_process = iterative_process_builder(tff_model_fn, client_weight_fn)

  client_datasets_fn = functools.partial(
      tff.simulation.build_uniform_sampling_fn(
          dataset=cifar_train.client_ids,
          random_seed=client_datasets_random_seed),
      size=clients_per_round)

  evaluate_fn = tff.learning.build_federated_evaluation(
      tff_model_fn, use_experimental_simulation_loop=True)

  def validation_fn(model_weights, round_num):
    del round_num
    return evaluate_fn(model_weights, [cifar_test])

  def test_fn(model_weights):
    return evaluate_fn(model_weights, [cifar_test])

  logging.info('Training model:')
  logging.info(model_builder().summary())

  training_loop.run(
      iterative_process=training_process,
      train_client_datasets_fn=client_datasets_fn,
      evaluation_fn=validation_fn,
      test_fn=test_fn,
      total_rounds=total_rounds,
      experiment_name=experiment_name,
      root_output_dir=root_output_dir,
      **kwargs)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')

  def iterative_process_builder(
      model_fn: Callable[[], tff.learning.Model],
      client_weight_fn: Optional[Callable[[Any], tf.Tensor]] = None,
  ) -> tff.templates.IterativeProcess:
    """Creates an iterative process using a given TFF `model_fn`.

    Args:
      model_fn: A no-arg function returning a `tff.learning.Model`.
      client_weight_fn: Optional function that takes the output of
        `model.report_local_outputs` and returns a tensor providing the weight
        in the federated average of model deltas. If not provided, the default
        is the total number of examples processed on device.

    Returns:
      A `tff.templates.IterativeProcess`.
    """

    return tff.learning.build_federated_averaging_process(
        model_fn=model_fn,
        client_optimizer_fn=client_optimizer_fn,
        server_optimizer_fn=server_optimizer_fn,
        client_weighting=client_weight_fn,
        use_experimental_simulation_loop=True)

  hparam_dict = _get_hparam_flags()

  run_federated(
      iterative_process_builder,
      num_classes=FLAGS.num_classes,
      client_epochs_per_round=FLAGS.client_epochs_per_round,
      client_batch_size=FLAGS.client_batch_size,
      clients_per_round=FLAGS.clients_per_round,
      client_datasets_random_seed=FLAGS.client_datasets_random_seed,
      crop_size=FLAGS.crop_size,
      total_rounds=FLAGS.total_rounds,
      uniform_weighting=FLAGS.uniform_weighting,
      experiment_name=FLAGS.experiment_name,
      root_output_dir=FLAGS.root_output_dir,
      rounds_per_eval=FLAGS.rounds_per_eval,
      rounds_per_checkpoint=FLAGS.rounds_per_checkpoint,
      hparam_dict=hparam_dict)


if __name__ == '__main__':
  app.run(main)
