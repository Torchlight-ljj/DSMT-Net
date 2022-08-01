from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, precision_score,f1_score,recall_score
from torch import Tensor
import torch
##write by myself
def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    class_dice = []
    for channel in range(input.shape[1]):
        dice_temp = dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)
        dice += dice_temp
        class_dice.append(dice_temp)

    return dice / input.shape[1] , class_dice

def compute_prec_recal(y_pred, y_true):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    # intersection = (y_true * y_pred).sum()
    # #intersection = np.sum(intersection)     
    # union = y_true.sum() + y_pred.sum() - intersection
    current = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = (tn+tp)/(tn+fp+fn+tp)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*precision*recall/(precision+recall)
    return acc,precision,recall,f1

def single_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (height, width)
    intersection = np.sum(y_true * y_pred_bin)
    if (np.sum(y_true)==0) and (np.sum(y_pred_bin)==0):
        return 1
    return (2*intersection) / (np.sum(y_true) + np.sum(y_pred_bin))

def mean_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (n_samples, height, width, n_channels)
    batch_size = y_true.shape[0]
    channel_num = y_true.shape[-1]
    mean_dice_channel = 0.
    for i in range(batch_size):
        for j in range(channel_num):
            channel_dice = single_dice_coef(y_true[i, :, :, j], y_pred_bin[i, :, :, j])
            mean_dice_channel += channel_dice/(channel_num*batch_size)
    return mean_dice_channel
# y_true = np.array([0,0,0,0,0,0,])
# y_pred = np.array([0,0,0,0,0,0,])
# print(compute_iou(y_pred,y_true))
# print(precision_score(y_true,y_pred),recall_score(y_true,y_pred),f1_score(y_true,y_pred))

def plot_matrix(matrix,save_name, labels_name, title=None, thresh=0.8, axis_labels=None):
    cm = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]  # 归一化

# 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Reds'))
    plt.colorbar()  # 绘制图例

# 图像标题
    if title is not None:
        plt.title(title)
# 绘制坐标
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    plt.xticks(num_local, axis_labels)  # 将标签印在x轴坐标上， 并倾斜45度
    plt.yticks(num_local, axis_labels)  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if int(cm[i][j] * 100 + 0.5) > 0:
                plt.text(j, i, format(int(cm[i][j] * 100 + 0.5), 'd') + '%',
                        ha="center", va="center",
                        color="white" if cm[i][j] > thresh else "black") 
    plt.savefig(save_name, transparent=True, dpi=800)   
    plt.close()

def compute_classify(preds,trues):
    true_class_num = [0,0,0,0]
    class_nums = 4
    imgs_nums = len(preds)
    confimg_matrix = np.zeros([class_nums,class_nums])
    acc = 0
    pred_class_num = [0,0,0,0]
    class0 = {'TP':0,'TN':0,'FP':0,'FN':0,'ALL':0,'Pres':0,'Recall':0,'F1':0}
    class1 = {'TP':0,'TN':0,'FP':0,'FN':0,'ALL':0,'Pres':0,'Recall':0,'F1':0}
    class2 = {'TP':0,'TN':0,'FP':0,'FN':0,'ALL':0,'Pres':0,'Recall':0,'F1':0}        
    class3 = {'TP':0,'TN':0,'FP':0,'FN':0,'ALL':0,'Pres':0,'Recall':0,'F1':0}

    for i in range(len(preds)):
        confimg_matrix[trues[i]][preds[i]] += 1
        if trues[i] == preds[i] == 0:
            class0['TP'] += 1
        if trues[i] == preds[i] == 1:
            class1['TP'] += 1
        if trues[i] == preds[i] == 2:
            class2['TP'] += 1
        if class_nums == 4:
            if trues[i] == preds[i] == 3:
                class3['TP'] += 1
        pred_class_num[int(preds[i])] += 1
        true_class_num[int(trues[i])] += 1
    class0['ALL'] = true_class_num[0]
    class1['ALL'] = true_class_num[1]
    class2['ALL'] = true_class_num[2]
    class3['ALL'] = true_class_num[3]
    if pred_class_num[0] == 0:
        class0['Pres'] = 0
    else:
        class0['Pres'] = round(class0['TP']/pred_class_num[0], 4)

    class0['Recall'] = round(class0['TP']/class0['ALL'], 4)

    if class0['Pres'] == 0 or class0['Recall']==0:
        class0['F1'] = 0
    else:
        class0['F1'] = round(2*class0['Pres']*class0['Recall']/(class0['Pres']+class0['Recall']),4)

    if pred_class_num[1] == 0:
        class1['Pres'] = 0
    else:
        class1['Pres'] = round(class1['TP']/pred_class_num[1], 4)
    class1['Recall'] = round(class1['TP']/class1['ALL'], 4)
    if class1['Pres'] == 0 or class1['Recall'] ==0 :
        class1['F1'] = 0
    else:
        class1['F1'] = round(2*class1['Pres']*class1['Recall']/(class1['Pres']+class1['Recall']),4)

    if pred_class_num[2] == 0:
        class2['Pres'] = 0
    else:
        class2['Pres'] = round(class2['TP']/pred_class_num[2], 4)
    class2['Recall'] = round(class2['TP']/class2['ALL'], 4)
    if class2['Pres'] == 0 or class2['Recall'] ==0 :
        class2['F1'] = 0
    else:
        class2['F1'] = round(2*class2['Pres']*class2['Recall']/(class2['Pres']+class2['Recall']),4)

    if pred_class_num[3] == 0:
        class3['Pres'] = 0
    else:
        class3['Pres'] = round(class3['TP']/pred_class_num[3], 4)
    class3['Recall'] = round(class3['TP']/class3['ALL'], 4)
    if class3['Recall'] == 0 or class3['Pres'] == 0:
        class3['F1'] = 0
    else:
        class3['F1'] = round(2*class3['Pres']*class3['Recall']/(class3['Pres']+class3['Recall']),4)

    acc = (class0['TP']+class1['TP']+class2['TP']+class3['TP']) /imgs_nums
    Presion = round((class0['Pres'] + class1['Pres']+ class2['Pres']+ class3['Pres'])/class_nums, 4)
    Recall =  round((class0['Recall'] + class1['Recall']+ class2['Recall']+ class3['Recall'])/class_nums, 4)
    F1 =  round((class0['F1'] + class1['F1']+ class2['F1']+ class3['F1'])/class_nums, 4)       

    return acc,Presion,Recall,F1
