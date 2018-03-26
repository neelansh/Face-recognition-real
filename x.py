from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from torch.autograd import Variable
from torch.autograd import Function
import torch.backends.cudnn as cudnn
import os
import numpy as np
from tqdm import tqdm
# from model import FaceModel,FaceModelCenter
from eval_metrics import evaluate
from logger import Logger
from LFWDataset import LFWDataset
from PIL import Image
from utils import PairwiseDistance,display_triplet_distance,display_triplet_distance_test
import collections



# Model options
args = {}
dataroot = '../test_align'
lfw_dir = '../lfw/lfw_aligned'
lfw_pairs_path='./lfw_pairs.txt'
log_dir='../log'
resume=False
resume_path='/media/lior/LinuxHDD/pytorch_face_logs/run-optim_adam-lr0.001-wd0.0-embeddings512-center0.5-MSCeleb/checkpoint_11.pth',
epochs=3
center_loss_weight=0.6
alpha=0.5
embedding_size=512
batch_size=64
test_batch_size=64
lr=0.001
beta1=0.5
lr_decay=1e-4
wd=0.0
optimizer='adam'
# Device options
no_cuda=False
gpu_id='0'
seed=0
log_interval=10

#args = parser.parse_args()

# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

args['cuda'] = not no_cuda and torch.cuda.is_available()
np.random.seed(seed)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if cuda:
    cudnn.benchmark = True

LOG_DIR = log_dir + '/run-optim_{}-lr{}-wd{}-embeddings{}-center{}-MSCeleb'.format(optimizer, lr, wd,embedding_size,center_loss_weight)

# create logger
# logger = Logger(LOG_DIR)

kwargs = {'num_workers': 2, 'pin_memory': True} if cuda else {}
l2_dist = PairwiseDistance(2)

transform = transforms.Compose([
                         transforms.Scale(96),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize(mean = [ 0.5, 0.5, 0.5 ],
                                               std = [ 0.5, 0.5, 0.5 ])
                     ])

train_dir = ImageFolder(dataroot,transform=transform)
train_loader = torch.utils.data.DataLoader(train_dir,
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    LFWDataset(dir=lfw_dir,pairs_path=lfw_pairs_path,
                     transform=transform),
    batch_size=batch_size, shuffle=False, **kwargs)




def main_loop():
    test_display_triplet_distance= True
    # print the experiment configuration
    # print('\nparsed options:\n{}\n'.format(vars(args))
    # print('\nNumber of Classes:\n{}\n'.format(len(train_dir.classes)))

    # instantiate model and initialize weights



    # optionally resume from a checkpoint
    checkpoint = None
    if resume:
        if os.path.isfile(resume_path):
            print('=> loading checkpoint {}'.format(resume_path))
            checkpoint = torch.load(resume_path)
            args['start_epoch'] = checkpoint['epoch']
        else:
            checkpoint = None
            print('=> no checkpoint found at {}'.format(resume_path))

    model = FaceModelCenter(embedding_size=embedding_size,num_classes=len(train_dir.classes)
                            ,checkpoint=checkpoint)

    if cuda:
        model.cuda()

    optimizer = create_optimizer(model, lr)

    start = start_epoch
    end = start + int(epochs)
    print(model)
    for epoch in range(start, end):
        train(train_loader, model, optimizer, epoch)
        test(test_loader, model, epoch)
        if test_display_triplet_distance:
            display_triplet_distance_test(model,test_loader,LOG_DIR+"/test_{}".format(epoch))



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(train_loader, model, optimizer, epoch):
    # switch to train mode
    model.train()

    pbar = tqdm(enumerate(train_loader))

    top1 = AverageMeter()

    for batch_idx, (data, label) in pbar:

        data_v = Variable(data.cuda())
        target_var = Variable(label)

        # compute output
        prediction = model.forward_classifier(data_v)

        center_loss, model.centers = model.get_center_loss(target_var, alpha)

        criterion = nn.CrossEntropyLoss()

        cross_entropy_loss = criterion(prediction.cuda(),target_var.cuda())

        loss = center_loss_weight*center_loss + cross_entropy_loss

        # compute gradient and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update the optimizer learning rate
        adjust_learning_rate(optimizer)

        # log loss value
        # logger.log_value('cross_entropy_loss', cross_entropy_loss.data[0]).step()
        logger.log_value('total_loss', loss.data[0]).step()

        prec = accuracy(prediction.data, label.cuda(), topk=(1,))
        top1.update(prec[0], data_v.size(0))

        if batch_idx % log_interval == 0:
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'
                'Train Prec@1 {:.2f} ({:.2f})'.format(
                    epoch, batch_idx * len(data_v), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.data[0],float(top1.val[0]), float(top1.avg[0])))



    logger.log_value('Train Prec@1 ',float(top1.avg[0]))

    # do checkpointing
    torch.save({'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'centers': model.centers},
               '{}/checkpoint_{}.pth'.format(LOG_DIR, epoch))


def test(test_loader, model, epoch):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:
        if cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()
        data_a, data_p, label = Variable(data_a, volatile=True), \
                                Variable(data_p, volatile=True), Variable(label)

        # compute output
        out_a, out_p = model(data_a), model(data_p)
        dists = l2_dist.forward(out_a,out_p)#torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        distances.append(dists.data.cpu().numpy())
        labels.append(label.data.cpu().numpy())

        if batch_idx % log_interval == 0:
            pbar.set_description('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data_a), len(test_loader.dataset),
                100. * batch_idx / len(test_loader)))

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist[0] for dist in distances for subdist in dist])

    tpr, fpr, accuracy, val, val_std, far = evaluate(distances,labels)
    print('\33[91mTest set: Accuracy: {:.8f}\n\33[0m'.format(np.mean(accuracy)))
    logger.log_value('Test Accuracy', np.mean(accuracy))

    plot_roc(fpr,tpr,figure_name="roc_test_epoch_{}.png".format(epoch))

def plot_roc(fpr,tpr,figure_name="roc.png"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    fig.savefig(os.path.join(LOG_DIR,figure_name), dpi=fig.dpi)


def adjust_learning_rate(optimizer):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0
        group['step'] += 1

        group['lr'] = lr / (1 + group['step'] * lr_decay)


def create_optimizer(model, new_lr):
    # setup optimizer
    if optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=wd)
    elif optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=wd, betas=(beta1, 0.999))
    elif optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=new_lr,
                                  lr_decay=lr_decay,
                                  weight_decay=wd)
    return optimizer