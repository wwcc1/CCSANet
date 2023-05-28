import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet12 import resnet12
from ccsam import CCSAM

def one_hot(labels_train):
    labels_train = labels_train.cpu()
    nKnovel = 5
    labels_train_1hot_size = list(labels_train.size()) + [nKnovel, ]
    labels_train_unsqueeze = labels_train.unsqueeze(dim=labels_train.dim())
    labels_train_1hot = torch.zeros(labels_train_1hot_size).scatter_(len(labels_train_1hot_size) - 1,
                                                                     labels_train_unsqueeze, 1)
    return labels_train_1hot


class Model(nn.Module):
    def __init__(self, scale_cls, iter_num_prob=35.0 / 75, num_classes=64):
        super(Model, self).__init__()
        self.scale_cls = scale_cls
        self.iter_num_prob = iter_num_prob
        self.base = resnet12()
        self.cam = CCSAM()
        self.nFeat = self.base.nFeat
        self.clasifier = nn.Conv2d(self.nFeat, num_classes, kernel_size=1)

    def test(self, ftrain, ftest):
        ftest = ftest.mean(4)
        ftest = ftest.mean(4)
        ftest = F.normalize(ftest, p=2, dim=ftest.dim() - 1, eps=1e-12)
        ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim() - 1, eps=1e-12)
        scores = self.scale_cls * torch.sum(ftest * ftrain, dim=-1)
        return scores

    def forward(self, xtrain, xtest, ytrain, ytest):

        batch_size, num_train = xtrain.size(0), xtrain.size(1)
        num_test = xtest.size(1)
        K = ytrain.size(2)
        ytrain = ytrain.transpose(1, 2)
        ytest6 = ytest.transpose(1, 2)
        xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))
        x = torch.cat((xtrain, xtest), 0)
        f = self.base(x)


        ftrain = f[:batch_size * num_train]
        ftrain6 = ftrain.view(batch_size, num_train, *f.size()[1:])
        ftrain = ftrain.view(batch_size, num_train, -1)

        ftrain = torch.bmm(ytrain, ftrain)
        ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))
        ftrain = ftrain.view(batch_size, -1, *f.size()[1:])

        fp2 = f[:batch_size * num_train].view(batch_size, -1, *f.size()[1:])
        ftest = f[batch_size * num_train:]

        dd  = ftest.view(batch_size, num_test, *f.size()[1:])

        ftest6 = ftest.view(batch_size, num_test, -1)
        ftest6 = torch.bmm(ytest6, ftest6)
        ftest6 = ftest6.div(ytest6.sum(dim=2, keepdim=True).expand_as(ftest6))
        ftest6 = ftest6.view(batch_size, -1, *f.size()[1:])
        ftest = ftest.view(batch_size, num_test, *f.size()[1:])

        ftrain, ftest, fp1, fp2, ftest6, ftrain6 = self.cam(ftrain, ftest, fp2, ftest6, ftrain6)
        ftrain = ftrain.mean(4)
        ftrain = ftrain.mean(4)
        fp1 = fp1.mean(4)
        fp1 = fp1.mean(4)
        ftest6 = ftest6.mean(4)
        ftest6 = ftest6.mean(4)

        if not self.training:
            return self.test(ftrain, ftest)

        ftest_norm = F.normalize(ftest, p=2, dim=3, eps=1e-12)
        ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
        ftrain_norm = ftrain_norm.unsqueeze(4)
        ftrain_norm = ftrain_norm.unsqueeze(5)
        cls_scores = self.scale_cls * torch.sum(ftest_norm * ftrain_norm, dim=3)
        cls_scores = cls_scores.view(batch_size * num_test, *cls_scores.size()[2:])

        fp2_norm = F.normalize(fp2, p=2, dim=3, eps=1e-12)
        fp1_norm = F.normalize(fp1, p=2, dim=3, eps=1e-12)
        fp1_norm = fp1_norm.unsqueeze(4)
        fp1_norm = fp1_norm.unsqueeze(5)
        cls_scores1 = self.scale_cls * torch.sum(fp2_norm * fp1_norm, dim=3)
        cls_scores1 = cls_scores1.view(batch_size * num_train, *cls_scores1.size()[2:])

        ftrain6_norm = F.normalize(ftrain6, p=2, dim=3, eps=1e-12)
        ftest6_norm = F.normalize(ftest6, p=2, dim=3, eps=1e-12)
        ftest6_norm = ftest6_norm.unsqueeze(4)
        ftest6_norm = ftest6_norm.unsqueeze(5)
        cls_scores2 = self.scale_cls * torch.sum(ftrain6_norm * ftest6_norm, dim=3)  # [4, 25, 5, 6, 6]
        cls_scores2 = cls_scores2.view(batch_size * num_train, *cls_scores2.size()[2:])  # [100, 5, 6, 6]

        ftrain6 = ftrain6.view(batch_size, num_train, K, -1)
        ftrain6 = ftrain6.transpose(2, 3)  # torch.Size([4, 25, 18432, 5])
        ytrain = ytrain.unsqueeze(3)  # torch.Size([4, 5, 25, 1])
        ytrain = ytrain.transpose(1, 2)
        ftrain6 = torch.matmul(ftrain6, ytrain)  # torch.Size([4, 25, 18432, 1])
        ftrain6 = ftrain6.view(batch_size * num_train, -1, 6, 6)
        ytrain = self.clasifier(ftrain6)  # torch.Size([100, 64, 6, 6])

        ftest = ftest.view(batch_size, num_test, K, -1)
        ftest = ftest.transpose(2, 3)  # torch.Size([4, 30, 18432, 5])
        ytest = ytest.unsqueeze(3)  # torch.Size([4, 30, 5, 1])
        ftest = torch.matmul(ftest, ytest)#torch.Size([4, 30, 18432, 1])
        ftest = ftest.view(batch_size * num_test, -1, 6,6)
        ytest = self.clasifier(ftest)
        dd   =dd.view(batch_size * num_test, -1, 6, 6)
        dd  =self.clasifier(dd )
        # aa  =self.clasifier(aa )
        return ytest, ytrain, cls_scores, cls_scores1, cls_scores2,dd#,aa

    def helper(self, ftrain, ftest, ytrain,fp2,ftest6,ftrain6):
        b, n, c, h, w = ftrain.size()#1 25
        k = ytrain.size(2)#5
        ytrain_transposed = ytrain.transpose(1, 2)#([1, 5, 25])
        ftrain = torch.bmm(ytrain_transposed, ftrain.view(b, n, -1))#([1, 5, 18432])
        ftrain = ftrain.div(ytrain_transposed.sum(dim=2, keepdim=True).expand_as(ftrain))#([1, 5, 18432])##求出每个类的原型
        ftrain = ftrain.view(b, -1, c, h, w)#([1, 5, 512, 6, 6])#相乘完后每一类的顺序不变
        aa = ftrain
        bb = ytrain_transposed
        ftrain, ftest,_,_,_,_ = self.cam(ftrain, ftest,fp2,ftest6,ftrain6)#([1, 75, 5, 512, 6, 6])
                                                                           #([1, 75, 5, 512, 6, 6])
        ftrain = ftrain.mean(-1).mean(-1)
        ftest = ftest.mean(-1).mean(-1)#([1, 75, 5, 512])
        #ftest.dim()指维数的个数4
        ftest = F.normalize(ftest, p=2, dim=ftest.dim() - 1, eps=1e-12)
        ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim() - 1, eps=1e-12)
        scores = self.scale_cls * torch.sum(ftest * ftrain, dim=-1)
        return scores ,aa, bb

    def test_transductive(self, xtrain, xtest, ytrain, ytest):




        iter_num_prob = self.iter_num_prob

        batch_size, num_train = xtrain.size(0), xtrain.size(1)#1 , 25
        num_test = xtest.size(1)#75
        K = ytrain.size(2)#5
        ytest6 = ytest.transpose(1, 2)#torch.Size([1, 5, 75])
        xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))#torch.Size([25, 3, 84, 84])
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))#torch.Size([75, 3, 84, 84])
        x = torch.cat((xtrain, xtest), 0)
        f = self.base(x)
        ftrain = f[:batch_size * num_train]#([25, 512, 6, 6])
        ftrain6 = ftrain.view(batch_size, num_train, *f.size()[1:])#([1, 25, 512, 6, 6])

        ftrain = f[: batch_size * num_train].view(batch_size, num_train, *f.size()[1:])#([1, 25, 512, 6, 6])
        fp2 = f[:batch_size * num_train].view(batch_size, -1, *f.size()[1:])#([1, 25, 512, 6, 6])
        ftest = f[batch_size * num_train:]#t([75, 512, 6, 6])
        ftest6 = ftest.view(batch_size, num_test, -1)#([1, 75, 18432])
        ftest6 = torch.bmm(ytest6, ftest6)  ##[4,5,25]  #([1, 5, 18432])
        ftest6 = ftest6.div(ytest6.sum(dim=2, keepdim=True).expand_as(ftest6))
        ftest6 = ftest6.view(batch_size, -1, *f.size()[1:])#torch.Size([1, 5, 512, 6, 6])
        ftest = f[batch_size * num_train:].view(batch_size, num_test, *f.size()[1:])#torch.Size([1, 75, 512, 6, 6])
        cls_scores ,aa,bb= self.helper(ftrain, ftest, ytrain,fp2,ftest6,ftrain6)#torch.Size([1, 75, 5])
        aa1,aa2,aa3,aa4,aa5= aa.size()
        bb1,bb2,bb3  = bb.size()
        num_images_per_iter = int(num_test * iter_num_prob)
        num_iter = num_test // num_images_per_iter

        for i in range(num_iter):
            max_scores, preds = torch.max(cls_scores, 2)
            chose_index = torch.argsort(max_scores.view(-1), descending=True)


            chose_index = chose_index[: num_images_per_iter * (i + 1)]
            s = ftest[0, chose_index]
            ftest_iter = ftest[0, chose_index].unsqueeze(0)


            s1,s2,s3,s4 = s.size()
            ftrainO = aa.view(aa1,aa2,-1)
            ftest_iterO =ftest_iter.view(1,s1,-1)
            p1,p2,p3,p4,p5 = torch.split(ftrainO, [1, 1, 1,1,1], dim=1)
            cosin1 = torch.cosine_similarity(p1, ftest_iterO, dim=2)*2
            cosin2 = torch.cosine_similarity(p2, ftest_iterO, dim=2)*2
            cosin3 = torch.cosine_similarity(p3, ftest_iterO, dim=2)*2
            cosin4 = torch.cosine_similarity(p4, ftest_iterO, dim=2)*2
            cosin5 = torch.cosine_similarity(p5, ftest_iterO, dim=2)*2
            pc =torch.cat((cosin1,cosin2,cosin3,cosin4,cosin5),0).unsqueeze(0)

            preds_iter = preds[0, chose_index].unsqueeze(0)
            preds_iter = one_hot(preds_iter).cuda()
            attention = preds_iter.transpose(1,2)*pc
            attention =torch.mean(attention*5, dim=1).unsqueeze(-1)
            ftest_iter1 = ftest_iterO*attention
            ftest_iter = ftest_iter1.view(-1,s1,s2,s3,s4)
            ftrain_iter = torch.cat((ftrain, ftest_iter), 1)
            ytrain_iter = torch.cat((ytrain, preds_iter), 1)

            cls_scores ,_ , _ = self.helper(ftrain_iter, ftest, ytrain_iter,fp2,ftest6,ftrain6)

        return cls_scores


