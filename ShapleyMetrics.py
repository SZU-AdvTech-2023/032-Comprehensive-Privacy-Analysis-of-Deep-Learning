import numpy as np
from math import factorial
from itertools import permutations
from ModelShokri import DataHandler, TrainWBAttacker
from torch.utils.data import DataLoader
import torch
import json
from utils import computeMetricsAlt, rescale01, computeMetrics
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm as tq

class ShapelyMetric():

  def __init__(self, players, args, evaluate_grad=True, loss_label=True, saved_path=None):
    self.players = players
    self.args = args
    self.evaluate_grad = evaluate_grad
    self.loss_label = loss_label
    self.saved_path = saved_path

    self.signal0, self.signal1 = self.load_signals()

    self.roc_utility_dict = {
      '': 0
    }
    self.acc_utility_dict = {
      '': 0
    }
    self.result_dict = {
       '':None
    }

  def load_signals(self):
    signal0 = np.load(self.args.signal0_path)
    signal1 = np.load(self.args.signal1_path)
    AdditionalInfo = np.load(self.args.additionalInfo_path)
    inter_outs0 = []
    inter_outs1 = []

    self.out_size_list = AdditionalInfo['arr_0']
    self.layer_size_list = AdditionalInfo['arr_1']
    self.kernel_size_list = AdditionalInfo['arr_2']

    n_inter_outputs = len(self.out_size_list)
    n_layer_grads = len(self.kernel_size_list)

    for i in range(n_inter_outputs):
      inter_outs0.append(signal0['arr_' + str(i)])
      inter_outs1.append(signal1['arr_' + str(i)])

    self.lossval0 = signal0['arr_' + str(n_inter_outputs)]
    self.lossval1 = signal1['arr_' + str(n_inter_outputs)]

    self.labels1hot0 = signal0['arr_' + str(n_inter_outputs + 1)]
    self.labels1hot1 = signal1['arr_' + str(n_inter_outputs + 1)]

    grad_vals0 = []
    grad_vals1 = []

    for i in range(n_inter_outputs + 2, n_inter_outputs + 2 + n_layer_grads, 1):
      grad_vals0.append(signal0['arr_' + str(i)])
      grad_vals1.append(signal1['arr_' + str(i)])

    self.data_size0 = self.lossval0.shape[0]
    self.data_size1 = self.lossval1.shape[0]

    if self.evaluate_grad:
      return grad_vals0, grad_vals1
    else:
      return [inter_outs0, grad_vals0], [inter_outs1, grad_vals1]
    
  def evaluate(self):
    N = len(self.players)
    self.roc_shapley_values = [0] * N
    self.acc_shapley_values = [0] * N
    index_mapping = {player: index for index, player in enumerate(self.players)}
    
    for perm in permutations(self.players):
        for i in range(N):
            coalition = perm[:i+1]
            contribution1 = self.coalition_value_func(coalition)
            contribution2 = self.coalition_value_func(perm[:i])
            self.roc_shapley_values[index_mapping[perm[i]]] += (contribution1[0] - contribution2[0])
            self.acc_shapley_values[index_mapping[perm[i]]] += (contribution1[1] - contribution2[1])
            

    # Normalize by dividing by N!
    normalization_factor = 1 / factorial(N)
    self.roc_shapley_values = [value * normalization_factor for value in self.roc_shapley_values]
    self.acc_shapley_values = [value * normalization_factor for value in self.acc_shapley_values]

    if self.saved_path is not None:
        save_dict = {
          'roc_svs': self.roc_shapley_values,
          'acc_svs': self.acc_shapley_values,
          'evaluate_result': self.result_dict
        }

        with open(self.saved_path, 'w') as json_file:
          json.dump(save_dict, json_file)
        print(f'results saved to {self.saved_path}')
    
    return self.roc_shapley_values, self.acc_shapley_values
  
  def coalition_value_func(self, coalition):
    sorted_coalition = sorted(coalition)
    key = ''.join(map(str, sorted_coalition))
    if key in self.roc_utility_dict:
      return self.roc_utility_dict[key], self.acc_utility_dict[key]
    else:
      if self.evaluate_grad:
        self.roc_utility_dict[key], self.acc_utility_dict[key] = self.analysis_Shokri_grad(coalition, key)
      else:
        self.roc_utility_dict[key], self.acc_utility_dict[key] = self.analysis_Shokri_signal(coalition, key)
    return self.roc_utility_dict[key], self.acc_utility_dict[key]

  def analysis_Shokri_grad(self, coalition, key):
    FPR = np.linspace(0, 1, num=1001)
    num_runs_for_random = self.args.num_iters
    aux_list_metrics = []
    aux_list_TPR = []
    for k in range(num_runs_for_random):
      np.random.seed(k)
      indx_train0 = np.random.choice(self.data_size0, size=self.args.train_num0, replace=False)
      indx_train1 = np.random.choice(self.data_size1, size=self.args.train_num1, replace=False)

      indx_test0 = np.setdiff1d(np.arange(self.data_size0), indx_train0)
      indx_test0 = np.random.choice(indx_test0, size=self.args.test_num0, replace=False)
      indx_test1 = np.setdiff1d(np.arange(self.data_size1), indx_train1)
      indx_test1 = np.random.choice(indx_test1, size=self.args.test_num1, replace=False)

      grad_val0 = [self.signal0[i] for i in coalition]
      grad_val1 = [self.signal1[i] for i in coalition]

      if self.loss_label:
          trainingData = DataHandler(grad0=grad_val0, grad1=grad_val1, loss0=self.lossval0, loss1=self.lossval1,\
                                     hot0=self.labels1hot0, hot1=self.labels1hot1, indices0=indx_train0, indices1=indx_train1)
          Max = trainingData.Max
          Min = trainingData.Min
          testingData = DataHandler(grad0=grad_val0, grad1=grad_val1, loss0=self.lossval0, loss1=self.lossval1, \
                                    hot0=self.labels1hot0, hot1=self.labels1hot1, indices0=indx_test0, indices1=indx_test1, Max=Max, Min=Min)
      else:
          trainingData = DataHandler(grad0=grad_val0, grad1=grad_val1, indices0=indx_train0, indices1=indx_train1)
          Max = trainingData.Max
          Min = trainingData.Min
          testingData = DataHandler(grad0=grad_val0, grad1=grad_val1, indices0=indx_test0, indices1=indx_test1, Max=Max, Min=Min)

      kernel_size_list = [self.kernel_size_list[i] for i in coalition]
      layer_size_list = [self.layer_size_list[i] for i in coalition]
      if self.loss_label:
          AttackerShokri = TrainWBAttacker(trainingData, testingData, [], layer_size_list, kernel_size_list, is_grad=True, is_label=True, is_loss=True)
      else:
          AttackerShokri = TrainWBAttacker(trainingData, testingData, [], layer_size_list, kernel_size_list, is_grad=True)

      dataloaderEval = DataLoader(testingData, batch_size=100, shuffle=False)
      scoresEval = []
      EvalY = []
      with torch.no_grad():
          for i, batch in enumerate(dataloaderEval):
              example = batch[0]
              target = batch[1]
              scoresEval.append(AttackerShokri(*example).detach())
              EvalY.append(target.cpu().data.numpy())
      scoresEval = torch.cat(scoresEval, axis=0)
      scoresEval = torch.squeeze(scoresEval)
      scoresEval = scoresEval.cpu().data.numpy()
      EvalY = np.squeeze(np.concatenate(EvalY, axis=0))

      TPR_, metrics_ = computeMetricsAlt(scoresEval, EvalY, FPR)
      aux_list_metrics.append(metrics_)
      aux_list_TPR.append(TPR_)

    metrics = np.stack(aux_list_metrics, 1)
    mean_metrics = np.mean(metrics, 1)
    std_metrics = np.std(metrics, 1)

    # save result
    result_row = {
          "Attack Strategy": 'Nasr White-Box',
          "is_grad": self.evaluate_grad,
          "is_loss_label": self.loss_label,
          'AUROC': mean_metrics[0], 'AUROC STD': std_metrics[0],
          'Best Accuracy': mean_metrics[1], 'Best Accuracy STD': std_metrics[1],
          'FPR at TPR80': mean_metrics[2], 'FPR at TPR80 STD': std_metrics[2],
          'FPR at TPR85': mean_metrics[3], 'FPR at TPR85 STD': std_metrics[3],
          'FPR at TPR90': mean_metrics[4], 'FPR at TPR90 STD': std_metrics[4],
          'FPR at TPR95': mean_metrics[5], 'FPR at TPR95 STD': std_metrics[5]}
    self.result_dict[key] = result_row
    
    # Return ROC, mean_metrics[1] is the best accuracy
    return mean_metrics[0], mean_metrics[1]


class ShapelyMetricForGradNorm():

  def __init__(self, players, args, overall=True, is_loss=True, saved_path=None):
    self.players = players
    self.args = args
    self.overall = overall
    self.is_loss = is_loss
    self.saved_path = saved_path

    self.load_signals()

    self.roc_utility_dict = {
      '': 0
    }
    self.acc_utility_dict = {
      '': 0
    }
    self.result_dict = {
       '':None
    }

  def load_signals(self):
    with open(self.args.signal_path, 'r') as json_file:
       signal = json.load(json_file)
    if self.overall:
      self.grad_norms0 = np.array(signal['non-members']['overall_norms'])
      self.grad_norms1 = np.array(signal['members']['overall_norms'])
    else:
      self.grad_norms0 = np.array(signal['non-members']['gradient_norms'])
      self.grad_norms1 = np.array(signal['members']['gradient_norms'])

    self.lossval0 = np.array(signal['non-members']['loss_val'])
    self.lossval1 = np.array(signal['members']['loss_val'])

    self.labels1hot0 = np.array(signal['non-members']['onehot'])
    self.labels1hot1 = np.array(signal['members']['onehot'])

    self.data_size0 = self.lossval0.shape[0]
    self.data_size1 = self.lossval1.shape[0]

  def evaluate_by_loss(self):
    FPR = np.linspace(0, 1, num=1001)
    num_runs_for_random = self.args.num_iters
    aux_list_metrics = []
    aux_list_TPR = []
    for k in tq(range(num_runs_for_random)):
      np.random.seed(k)
      
      # Evaluating Set
      indx_eval0 = np.random.choice(self.lossval0.shape[0], size=self.args.test_num0, replace=False)
      indx_eval1 = np.random.choice(self.lossval1.shape[0], size=self.args.test_num1, replace=False)

      indx_diff0 = np.setdiff1d(np.arange(self.lossval0.shape[0]), indx_eval0)
      indx_diff1 = np.setdiff1d(np.arange(self.lossval1.shape[0]), indx_eval1)

      # Training Set (When needed)
      indx_training0 = np.random.choice(indx_diff0, size=self.args.train_num0, replace=False)
      indx_training1 = np.random.choice(indx_diff1, size=self.args.train_num1, replace=False)

      evalX0 = self.lossval0[indx_eval0].reshape(-1, 1)
      trainX0 = self.lossval0[indx_training0].reshape(-1, 1)
      evalX1 = self.lossval1[indx_eval1].reshape(-1, 1)
      trainX1 = self.lossval1[indx_training1].reshape(-1, 1)

      trainY0 = np.zeros((trainX0.shape[0]))
      trainY1 = np.ones((trainX1.shape[0]))

      trainX = np.concatenate((trainX0, trainX1), 0)
      trainY = np.concatenate((trainY0, trainY1))

      Max = np.max(trainX, axis=0)
      Min = np.min(trainX, axis=0)

      trainX = rescale01(trainX, Max, Min)
      evalX0 = rescale01(evalX0, Max, Min)
      evalX1 = rescale01(evalX1, Max, Min)

      attackModelRezaei = LogisticRegression(penalty='l2', tol=1e-5, random_state=k, solver='saga', max_iter=150)
      attackModelRezaei.fit(trainX, trainY)
      preds0 = attackModelRezaei.predict_proba(evalX0)[:, 1]
      preds1 = attackModelRezaei.predict_proba(evalX1)[:, 1]

      TPR_, metrics_ = computeMetrics(preds0, preds1, FPR)
      aux_list_metrics.append(metrics_)
      aux_list_TPR.append(TPR_)

    metrics = np.stack(aux_list_metrics, 1)
    mean_metrics = np.mean(metrics, 1)
    print(f'finished evaluated loss --> roc: {mean_metrics[0] * 100:.2f}%, acc: {mean_metrics[1] * 100:.2f}%')

  def evaluate(self):
    N = len(self.players)
    self.roc_shapley_values = [0] * N
    self.acc_shapley_values = [0] * N
    index_mapping = {player: index for index, player in enumerate(self.players)}
    
    for perm in permutations(self.players):
        for i in range(N):
            coalition = perm[:i+1]
            contribution1 = self.coalition_value_func(coalition)
            contribution2 = self.coalition_value_func(perm[:i])
            self.roc_shapley_values[index_mapping[perm[i]]] += (contribution1[0] - contribution2[0])
            self.acc_shapley_values[index_mapping[perm[i]]] += (contribution1[1] - contribution2[1])
            
    # Normalize by dividing by N!
    normalization_factor = 1 / factorial(N)
    self.roc_shapley_values = [value * normalization_factor for value in self.roc_shapley_values]
    self.acc_shapley_values = [value * normalization_factor for value in self.acc_shapley_values]

    if self.saved_path is not None:
        save_dict = {
          'roc_svs': self.roc_shapley_values,
          'acc_svs': self.acc_shapley_values,
          'evaluate_result': self.result_dict
        }

        with open(self.saved_path, 'w') as json_file:
          json.dump(save_dict, json_file)
        print(f'results saved to {self.saved_path}')
    
    return self.roc_shapley_values, self.acc_shapley_values
  
  def coalition_value_func(self, coalition):
    sorted_coalition = sorted(coalition)
    key = ','.join(map(str, sorted_coalition))
    if key in self.roc_utility_dict:
      return self.roc_utility_dict[key], self.acc_utility_dict[key]
    else:
        self.roc_utility_dict[key], self.acc_utility_dict[key] = self.analysis_gradnorm(coalition, key)
    return self.roc_utility_dict[key], self.acc_utility_dict[key]

  def analysis_gradnorm(self, coalition, key):
    FPR = np.linspace(0, 1, num=1001)
    num_runs_for_random = self.args.num_iters
    aux_list_metrics = []
    aux_list_TPR = []
    for k in tq(range(num_runs_for_random)):
      np.random.seed(k)
      
      # Evaluating Set
      indx_eval0 = np.random.choice(self.grad_norms0.shape[0], size=self.args.test_num0, replace=False)
      indx_eval1 = np.random.choice(self.grad_norms1.shape[0], size=self.args.test_num1, replace=False)

      indx_diff0 = np.setdiff1d(np.arange(self.grad_norms0.shape[0]), indx_eval0)
      indx_diff1 = np.setdiff1d(np.arange(self.grad_norms1.shape[0]), indx_eval1)

      # Training Set (When needed)
      indx_training0 = np.random.choice(indx_diff0, size=self.args.train_num0, replace=False)
      indx_training1 = np.random.choice(indx_diff1, size=self.args.train_num1, replace=False)

      evalX0 = self.grad_norms0[indx_eval0][:, coalition].reshape(self.args.test_num0, -1)
      trainX0 = self.grad_norms0[indx_training0][:, coalition].reshape(self.args.train_num0, -1)
      evalX1 = self.grad_norms1[indx_eval1][:, coalition].reshape(self.args.test_num1, -1)
      trainX1 = self.grad_norms1[indx_training1][:, coalition].reshape(self.args.train_num1, -1)

      if self.is_loss:
         evalX0 = np.concatenate((evalX0, self.lossval0[indx_eval0][:, np.newaxis]), axis=1)
         trainX0 = np.concatenate((trainX0, self.lossval0[indx_training0][:, np.newaxis]), axis=1)
         evalX1 = np.concatenate((evalX1, self.lossval1[indx_eval1][:, np.newaxis]), axis=1)
         trainX1 = np.concatenate((trainX1, self.lossval1[indx_training1][:, np.newaxis]), axis=1)

      trainY0 = np.zeros((trainX0.shape[0]))
      trainY1 = np.ones((trainX1.shape[0]))

      trainX = np.concatenate((trainX0, trainX1), 0)
      trainY = np.concatenate((trainY0, trainY1))

      Max = np.max(trainX, axis=0)
      Min = np.min(trainX, axis=0)

      trainX = rescale01(trainX, Max, Min)
      evalX0 = rescale01(evalX0, Max, Min)
      evalX1 = rescale01(evalX1, Max, Min)

      attackModelRezaei = LogisticRegression(penalty='l2', tol=1e-4, random_state=k, solver='saga', max_iter=200)
      attackModelRezaei.fit(trainX, trainY)
      preds0 = attackModelRezaei.predict_proba(evalX0)[:, 1]
      preds1 = attackModelRezaei.predict_proba(evalX1)[:, 1]

      TPR_, metrics_ = computeMetrics(preds0, preds1, FPR)
      aux_list_metrics.append(metrics_)
      aux_list_TPR.append(TPR_)

    metrics = np.stack(aux_list_metrics, 1)
    mean_metrics = np.mean(metrics, 1)
    std_metrics = np.std(metrics, 1)

    # save result
    result_row = {
          "Attack Strategy": 'Gradient Norms',
          "is_overall": self.overall,
          "is_loss": self.is_loss,
          'AUROC': mean_metrics[0], 'AUROC STD': std_metrics[0],
          'Best Accuracy': mean_metrics[1], 'Best Accuracy STD': std_metrics[1],
          'FPR at TPR80': mean_metrics[2], 'FPR at TPR80 STD': std_metrics[2],
          'FPR at TPR85': mean_metrics[3], 'FPR at TPR85 STD': std_metrics[3],
          'FPR at TPR90': mean_metrics[4], 'FPR at TPR90 STD': std_metrics[4],
          'FPR at TPR95': mean_metrics[5], 'FPR at TPR95 STD': std_metrics[5]}
    self.result_dict[key] = result_row
    
    print(f'finished evaluated coalition {key} --> roc: {mean_metrics[0] * 100:.2f}%, roc std: {std_metrics[0] * 100:.2f}%, acc: {mean_metrics[1] * 100:.2f}%, acc std: {std_metrics[1] * 100:.2f}%')
    # Return ROC, mean_metrics[1] is the best accuracy
    return mean_metrics[0], mean_metrics[1]
  