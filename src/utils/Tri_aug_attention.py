import torch
from torch import nn
import math
from transformers import BertConfig
class Tri_aug_attlayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.opt = dict(
            EThidden_size=args.d_prjh,
            multihead=args.multi_head,
            ETlayers=args.ETlayers
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_config = BertConfig(
                    hidden_size=self.opt['EThidden_size'],
                    num_attention_heads=self.opt['multihead'],
                    intermediate_size=self.opt['EThidden_size'] * 4,
                    hidden_dropout_prob=0.3,
                    attention_probs_dropout_prob=0.3,
        )
        self.cross_modal_layers1 = nn.ModuleList([BertCrossLayer(self.bert_config) for _ in range(self.opt['ETlayers'])])
        self.cross_modal_layers2 = nn.ModuleList([BertCrossLayer(self.bert_config) for _ in range(self.opt['ETlayers'])])
    def forward(
        self,
        lang_emb,
        img_emb,
        ):
        x = lang_emb
        y = img_emb
        text_masks = torch.zeros([x.shape[0], 1, 1, x.shape[1]], dtype=torch.bool,device=lang_emb.device)
        extend_image_masks = torch.ones((y.size(0),1,1,y.size(1)), dtype=torch.long, device=self.device)
        link_layer_index = 0
        last_layer=False
        for i in range(self.opt['ETlayers']):
            if i == self.opt['ETlayers'] - 1:
                last_layer=True
            image_textaug_embeds= self.cross_modal_layers1[link_layer_index](x, y, text_masks,extend_image_masks,last_layer)
            text_imageaug_embeds= self.cross_modal_layers2[link_layer_index](y, x,extend_image_masks, text_masks,last_layer)
            x = text_imageaug_embeds
            y = image_textaug_embeds
            link_layer_index += 1
        text_feats, image_feats= x, y
        return text_feats, image_feats, text_masks, extend_image_masks