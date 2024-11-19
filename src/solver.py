from torch import nn
import sys
import datetime
import time
from utils.eval_metrics import *
from utils.tools import *
from MSA import MSA
from transformers import get_linear_schedule_with_warmup
from src.utils.Sinkhorn import CustomMultiLossLayer
class Solver(object):
    def __init__(self, hyp_params, train_loader, dev_loader, test_loader, is_train=True, model=None,
                 pretrained_emb=None):
        self.hp = hp = hyp_params  # args
        self.epoch_i = 0
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader

        self.is_train = is_train
        self.model = model

        # initialize the model
        if model is None:
            self.model = model = MSA(hp)  # 本文的模型在这里

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            model = model.cuda()
        else:
            self.device = torch.device("cpu")

        # optimizer
        self.optimizer = {}

        if self.is_train:
            main_param = []
            bert_param = []
            encoder_tower_params = []
            TimeEncoder_params = []

            # 先收集所有需要的参数
            for name, p in model.named_parameters():  # 遍历输出参数
                if p.requires_grad:
                    if 'bert' in name:
                        bert_param.append(p)
                    elif 'ET_TA' in name or 'ET_TV' in name or 'ET_AV' in name:
                        encoder_tower_params.append(p)
                    elif 'a_encoder' in name or 'v_encoder' in name:
                        TimeEncoder_params.append(p)
                    else:
                        main_param.append(p)
            # 对main_param中的参数进行初始化
            for p in main_param:
                if p.dim() > 1:
                    nn.init.kaiming_normal_(p, mode='fan_in')
            for p in encoder_tower_params:
                if p.dim() > 1:
                    nn.init.kaiming_normal_(p, mode='fan_in')
            for p in TimeEncoder_params:
                if p.dim() > 1:
                    nn.init.kaiming_normal_(p, mode='fan_in')
            # 创建优化器参数组
            self.optimizer_main_group = [
                {'params': bert_param, 'weight_decay': hp.weight_decay_bert, 'lr': hp.lr_bert},
                {'params': main_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main},
                {'params': encoder_tower_params, 'weight_decay': hp.weight_decay_et, 'lr': hp.lr_et},
                {'params': TimeEncoder_params, 'weight_decay': hp.weight_decay_te, 'lr': hp.lr_te},
            ]

            self.optimizer = getattr(torch.optim, self.hp.optim)(
                self.optimizer_main_group
            )
            num_training_steps = int(self.hp.n_train / self.hp.batch_size * self.hp.num_epochs)
            warmup_steps = int(self.hp.warmup_ratio * num_training_steps)

            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
            )
    ####################################################################
    #
    # Training and evaluation scripts
    #
    ####################################################################

    def train_and_eval(self):
        model = self.model
        optimizer = self.optimizer
        print("Learning Rates for Each Parameter Group:")
        print(f"BERT learning rate: {self.hp.lr_bert}")
        print(f"Main learning rate: {self.hp.lr_main}")
        print(f"Encoder Tower learning rate: {self.hp.lr_et}")
        print()
        scheduler = self.scheduler
        custom_loss_layer = CustomMultiLossLayer(loss_num=4, device=self.device)
        def train(model, optimizer, scheduler):
            epoch_loss = 0

            model.train()
            num_batches = self.hp.n_train // self.hp.batch_size
            s_loss, fusion_loss,proc_size,recon_loss, dis_loss,avg_loss1, avg_loss2,avg_loss3,avg_loss4 = 0, 0, 0, 0, 0, 0,0,0,0
            start_time = time.time()
            a_size = self.hp.a_size
            v_size = self.hp.v_size
            for i_batch, batch_data in enumerate(self.train_loader):
                visual, vlens, audio, alens, r_labels,c_labels,l, bert_sent, bert_sent_mask, ids = batch_data

                model.zero_grad()

                with torch.cuda.device(0):
                    visual, audio, r_labels,c_labels, l, bert_sent, bert_sent_mask = \
                        visual.cuda(), audio.cuda(), r_labels.cuda(), c_labels.cuda(),l.cuda(), bert_sent.cuda(), \
                            bert_sent_mask.cuda()

                batch_size = r_labels.size(0)

                r_preds, r_preds_F, rec_loss,dis_loss = model(visual, audio, v_size, a_size, bert_sent, bert_sent_mask)

                h_loss = nn.L1Loss()

                single_loss = h_loss(r_preds, r_labels)
                fusion_loss = h_loss(r_preds_F, r_labels)

                #如果是分类任务，使用交叉熵损失
                # criterion_cls = nn.CrossEntropyLoss()
                # classification_loss = criterion_cls(c_preds, c_labels)
                # 使用自定义损失层计算总损失
                loss = custom_loss_layer([single_loss,fusion_loss, rec_loss,dis_loss])
                # loss = custom_loss_layer([single_loss, rec_loss, dis_loss])
                # loss = y_loss + classification_loss + nce_loss / 6
                # loss = single_loss + nce_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.hp.clip)
                optimizer.step()
                scheduler.step()

                s_loss += single_loss.item() * batch_size
                fusion_loss += fusion_loss.item() * batch_size
                recon_loss += rec_loss * batch_size
                dis_loss += dis_loss * batch_size
                proc_size += batch_size
                epoch_loss += loss.item() * batch_size


                if i_batch % 10 == 0 and i_batch > 0:
                    avg_loss1 = s_loss / proc_size
                    avg_loss2 = fusion_loss / proc_size
                    avg_loss3 = recon_loss/ proc_size
                    avg_loss4 = dis_loss/ proc_size
                    elapsed_time = time.time() - start_time
                    print(
                        'Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | S_loss  {:5.4f} | F_loss {:5.4f} | REC {:5.4f}| DIS {:5.4f}'.
                        format(epoch, i_batch, num_batches, elapsed_time * 1000 / self.hp.log_interval, avg_loss1,
                               avg_loss2,avg_loss3,avg_loss4))
                    s_loss, fusion_loss,recon_loss,dis_loss,proc_size = 0, 0, 0,0,0
                    start_time = time.time()

            return epoch_loss / self.hp.n_train

        def evaluate(model, test=False):
            model.eval()
            loader = self.test_loader if test else self.dev_loader
            total_loss = 0.0

            results = []
            truths = []

            with torch.no_grad():
                for i_batch, batch_data in enumerate(loader):
                    visual, vlens, audio, alens, r_labels,c_labels,lengths, bert_sent, bert_sent_mask, ids = batch_data

                    with torch.cuda.device(0):
                        audio, visual, r_labels,c_labels = audio.cuda(), visual.cuda(), r_labels.cuda(), c_labels.cuda()
                        # print(visual.size())
                        lengths = lengths.cuda()
                        bert_sent, bert_sent_mask = bert_sent.cuda(), bert_sent_mask.cuda()
                    batch_size = lengths.size(0)  # bert_sent in size (bs, seq_len, emb_size)

                    r_preds,r_preds_F,recon_loss,dis_loss = model(visual, audio, vlens, alens, bert_sent, bert_sent_mask)

                    # criterion = nn.SmoothL1Loss()
                    criterion = nn.L1Loss()
                    # criterion_cls = nn.CrossEntropyLoss()
                    s_loss = criterion(r_preds, r_labels)
                    fusion_loss = criterion(r_preds_F, r_labels)
                    # classification_loss = criterion_cls(c_preds, c_labels)
                    total_loss += (s_loss.item()+fusion_loss.item()+recon_loss+dis_loss)*batch_size
                    # total_loss += (s_loss.item()+recon_loss+dis_loss)*batch_size
                    res = (r_preds_F+r_preds)/2
                    results.append(res)
                    # results.append(r_preds)
                    truths.append(r_labels)

            avg_loss = total_loss / (self.hp.n_test if test else self.hp.n_valid)

            results = torch.cat(results)
            truths = torch.cat(truths)
            return avg_loss, results, truths
        best_accu = 1e-8
        best_mae = 1e8
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_filename = f"log/training_log_{current_time}.txt"
        with open(log_filename, 'w') as log_file:
            for epoch in range(1, self.hp.num_epochs + 1):
                start = time.time()

                # minimize all losses left
                train_loss = train(model, optimizer, scheduler)

                val_loss, _, _ = evaluate(model, test=False)
                test_loss, results, truths = evaluate(model, test=True)

                end = time.time()
                duration = end - start
                # scheduler.step(val_loss)

                epoch_info = f"Epoch {epoch} | Time {duration:.4f} sec | Train Loss {train_loss:.4f} | Valid Loss {val_loss:.4f} | Test Loss {test_loss:.4f}\n"
                log_file.write(epoch_info)
                print(epoch_info)

                if self.hp.dataset in ["mosei_senti", "mosei"]:
                    accu, mae, res_dict = eval_mosei_senti(results, truths, True)
                elif self.hp.dataset == 'mosi':
                    accu, mae, res_dict = eval_mosei_senti(results, truths, True)
                print(f'accu: {accu}')
                print(f'best_accu: {best_accu}')
                print(f'mae: {mae}')
                print(f'best_mae: {best_mae}')
                if mae <= best_mae:
                    best_mae = mae
                    best_epoch = epoch
                    best_results = results
                    best_truths = truths
                    print(f"Saved model at pre_trained_models of best_mae/MM.pt!")
                    save_model(self.hp, model, type='mae')

                if accu >= best_accu:
                    best_accu = accu
                    best_epoch = epoch
                    best_results = results
                    best_truths = truths
                    print(f"Saved model at pre_trained_models of best_acc/MM.pt!")
                    save_model(self.hp, model, type='acc')

                # 将每个epoch的评估结果写入文件
                if self.hp.dataset in ["mosi", "mosei"]:
                    accu, mae, eval_results = eval_mosei_senti(results, truths, False)
                    print('log generated')
                    log_file.write(f"BERT learning rate = {self.hp.lr_bert}\n")
                    log_file.write(f"Main learning rate = {self.hp.lr_main}\n")
                    log_file.write(f"Encoder Tower learning rate = {self.hp.lr_et}\n")
                    log_file.write(f"TSE learning rate = {self.hp.lr_te}\n")
                    log_file.write(f"a_size = {self.hp.a_size}\n")
                    log_file.write(f"v_size = {self.hp.v_size}\n")
                    log_file.write(f"step_ratio = {self.hp.step_ratio}\n")
                    log_file.write(f"TSElayer = {self.hp.tse_layers}\n")
                    log_file.write(f"ETlayer = {self.hp.ETlayers}\n")
                    log_file.write(f"weight_decay_te = {self.hp.weight_decay_te}\n")
                    for key, value in eval_results.items():
                        log_file.write(f"Epoch {epoch + 1}: {key} = {value}\n")
                    print('log wrote')
                log_file.write("-" * 50 + "\n")

        print(f'Best epoch: {best_epoch}')
        if self.hp.dataset in ["mosei_senti", "mosei"]:
            eval_mosei_senti(best_results, best_truths, True)
        elif self.hp.dataset == 'mosi':
            eval_mosei_senti(best_results, best_truths, True)
        sys.stdout.flush()