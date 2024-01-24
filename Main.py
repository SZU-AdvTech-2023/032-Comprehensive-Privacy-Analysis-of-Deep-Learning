import argparse
from ShapleyMetrics import ShapelyMetric, ShapelyMetricForGradNorm

def metric_gradients():
  parser = argparse.ArgumentParser(description='Analyse criteria obtained from different MIAs.')

  parser.add_argument('--num_iters', type=int, default=1, help='Number of iterations for empirical estimation.')

  # parser.add_argument('--signal0_path', type=str, default='./RawResults/CIFAR100/NasrTrain0_CNN_cifar100.npz', help='signal0_path.')
  # parser.add_argument('--signal1_path', type=str, default='./RawResults/CIFAR100/NasrTrain1_CNN_cifar100.npz', help='signal1_path.')
  # parser.add_argument('--additionalInfo_path', type=str, default='./RawResults/CIFAR100/NasrAddInfo_CNN_cifar100.npz', help='additionalInfo_path.')

  parser.add_argument('--signal0_path', type=str, default='./RawResults/NasrTrain0_AlexNet_loss_label.npz', help='signal0_path.')
  parser.add_argument('--signal1_path', type=str, default='./RawResults/NasrTrain1_AlexNet_loss_label.npz', help='signal1_path.')
  parser.add_argument('--additionalInfo_path', type=str, default='./RawResults/NasrAddInfo_AlexNet_loss_label.npz', help='additionalInfo_path.')

  parser.add_argument('--train_num0', type=int, default=5000, help='train_num0.')
  parser.add_argument('--train_num1', type=int, default=10000, help='train_num1.')
  parser.add_argument('--test_num0', type=int, default=5000, help='test_num0.')
  parser.add_argument('--test_num1', type=int, default=5000, help='test_num1.')

  args = parser.parse_args()
  saved_path = './SHAP_Result/AlexNet_CIFAR100/grad.json'
  shapely_metric = ShapelyMetric([3], args, evaluate_grad=True, loss_label=False, saved_path=saved_path)
  rocsvs, accsvs = shapely_metric.evaluate()
  print(rocsvs)
  print(accsvs)

def metric_gradnorms():
  parser = argparse.ArgumentParser(description='Analyse criteria obtained from different MIAs.')

  parser.add_argument('--num_iters', type=int, default=20, help='Number of iterations for empirical estimation.')
  parser.add_argument('--signal_path', type=str, default='./RawResults/Grad_Norm/grad_norms_AlexNet_CIFAR100.json', help='signal_path.')

  parser.add_argument('--train_num0', type=int, default=5000, help='train_num0.')
  parser.add_argument('--train_num1', type=int, default=15000, help='train_num1.')
  parser.add_argument('--test_num0', type=int, default=5000, help='test_num0.')
  parser.add_argument('--test_num1', type=int, default=5000, help='test_num1.')

  args = parser.parse_args()
  saved_path = './SHAP_Result/Grad_Norms/alexnet_cifar100.json'
  shapely_metric = ShapelyMetricForGradNorm([0], args, overall=False, is_loss=False, saved_path=saved_path)
  rocsvs, accsvs = shapely_metric.evaluate()
  print(rocsvs)
  print(accsvs)

# metric_gradients()
metric_gradnorms()