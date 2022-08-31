import tensorflow as tf
import gpflow
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyarrow
from gpflow.ci_utils import ci_niter
from tqdm import tqdm

print(tf.config.list_physical_devices('GPU'))


import argparse
import pandas as pd


parser = argparse.ArgumentParser(description='Model the traffic flow')

parser.add_argument('--makedata', dest='makedata', action='store_const',
                    const=True, default=False,
                    help='make the median data (default: assume the data is already made)')

args = parser.parse_args()


if args.makedata:
  # Import count data
  clean_birmingham_report_df_norm = pd.read_feather('high_quality_traffic_reports/clean_birmingham_report_df_norm')

  # For each timetstamp take the median normalised count value
  median_birmingham_report_df_norm = clean_birmingham_report_df_norm.groupby('timestamp')['total_volume_normalised'].median().to_frame().reset_index()
  # median_birmingham_report_df_norm['float_time'] = median_birmingham_report_df_norm.timestamp.astype(int)/1E9
  median_birmingham_report_df_norm['float_time'] = (median_birmingham_report_df_norm.timestamp.apply(lambda x: x.value) - 1552954440000000000)/1E9 * 1/900
  
  # Save the dataset
  median_birmingham_report_df_norm.to_feather('high_quality_traffic_reports/median_birmingham_report_df_norm')
  print('Generated the median data')
  
else:
  median_birmingham_report_df_norm = pd.read_feather('high_quality_traffic_reports/median_birmingham_report_df_norm')
  print('Imported the median data')
  

# Model normalised traffic as a gausian process
print('\nX and Y data\n')
X = np.array(median_birmingham_report_df_norm.float_time).reshape(-1,1)
Y = np.array(median_birmingham_report_df_norm.total_volume_normalised).reshape(-1,1)
Y_mean = Y.mean()
Y = Y - Y_mean
N = len(X)
train_dataset = tf.data.Dataset.from_tensor_slices((X, Y)).repeat().shuffle(N)

# Intialise GP
M = 10000  # Number of inducing locations
print(f'\nPicking M = {M} inducing points for our inital GP\n')
Z = np.random.choice(X.reshape(-1),M).reshape(-1,1)
kernel = gpflow.kernels.SquaredExponential()
m = gpflow.models.SVGP(kernel, gpflow.likelihoods.Gaussian(), Z, num_data=N)

# We turn off training for inducing point locations
gpflow.set_trainable(m.inducing_variable, False)


# Model training function
def run_adam(model, iterations, minibatch_size=1000):
  """
  Utility function running the Adam optimizer
  :param model: GPflow model
  :param interations: number of iterations
  """
  # Create an Adam Optimizer action
  logf = []
  train_iter = iter(train_dataset.batch(minibatch_size))
  training_loss = model.training_loss_closure(train_iter, compile=True)
  optimizer = tf.optimizers.Adam()

  @tf.function
  def optimization_step():
    optimizer.minimize(training_loss, model.trainable_variables)

  for step in tqdm(range(iterations)):
    optimization_step()
    if step % 10 == 0:
      elbo = -training_loss().numpy()
      logf.append(elbo)
  return logf


# Plotting function
def plot(title=""):
  plt.figure(figsize=(240, 4))
  plt.title(title)
  pX = np.linspace(X.min(), X.max(), 100)[:, None]  # Test locations
  pY, pYv = m.predict_y(pX)  # Predict Y values at test locations
  plt.plot(X, Y, "x", label="Training points", alpha=0.2)
  (line,) = plt.plot(pX, pY, lw=1.5, label="Mean of predictive posterior")
  col = line.get_color()
  plt.fill_between(
    pX[:, 0],
    (pY - 2 * pYv ** 0.5)[:, 0],
    (pY + 2 * pYv ** 0.5)[:, 0],
    color=col,
    alpha=0.6,
    lw=1.5,
  )
  Z = m.inducing_variable.Z.numpy()
  plt.plot(Z, np.zeros_like(Z), "k|", mew=2, label="Inducing locations")
  plt.legend(loc="lower right")
  plt.savefig(title + ".png")    

print('\nAttempting initial plot\n')
plot(title="Predictions before training")


print('\nTraining on mini batches\n')
# Train on mini batches
maxiter = ci_niter(100)
logf = run_adam(m, maxiter, 1000)

plot("Predictions after training")