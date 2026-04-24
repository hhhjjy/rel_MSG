import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet18, convnext_tiny, convnext_small, convnext_base
from transformers import Dinov2Model, ViTModel


class ResNetEmbedder(nn.Module):
    def __init__(self, model_type, output_type='vec', freeze=True, weights="DEFAULT"):
        super().__init__()
        self.backbone = None
        if model_type == 'resnet50':
            self.backbone = resnet50(weights=weights)
        elif model_type == 'resnet18':
            self.backbone = resnet18(weights=weights)
        else:
            raise NotImplementedError
        
        self.feature_dim = int(self.backbone.fc.in_features)
        
        if output_type == 'vec':
            self.backbone.fc = nn.Identity()
        elif output_type == 'feature':
            # Remove the avgpool and fc layer to output the feature map
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-3])
        else:
            raise NotImplementedError

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def forward(self, pixel_values):
        return self.backbone(pixel_values)


class ConvNextEmbedder(nn.Module):
    def __init__(self, model_type, output_type='vec', freeze=True, weights='DEFAULT'):
        super().__init__()
        if model_type == "convnext-tiny-224":
            self.backbone = convnext_tiny(weights=weights)
        elif model_type == "convnext-small-224":
            self.backbone = convnext_small(weights=weights)
        elif model_type == "convnext-base-224":
            self.backbone = convnext_base(weights=weights)
        else:
            raise NotImplementedError
        
        self.feature_dim = int(self.backbone.classifier[2].in_features)

        if output_type == "vec":
            self.backbone.classifier[2] = nn.Identity()
        elif output_type == "feature":
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            self.backbone[-1] = nn.Sequential(*list(self.backbone[-1].children())[:-1]) 
        else:
            raise NotImplementedError 

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def forward(self, pixel_values):
        return self.backbone(pixel_values)


class ViTEmbedder(nn.Module):
    def __init__(self, model_type, output_type='mean', freeze=True, weights="DEFAULT"):
        super().__init__()
        if model_type == "vit_base":
            self.vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
        elif model_type == "vit_large":
            self.vit_model = ViTModel.from_pretrained("google/vit-large-patch16-224")
        elif model_type == "vit_huge":
            self.vit_model = ViTModel.from_pretrained("google/vit-huge-patch14-224")
        else:
            raise NotImplementedError(f"Model type {model_type} is not implemented")

        if freeze:
            for param in self.vit_model.parameters():
                param.requires_grad = False
            self.vit_model.eval()
        
        self.feature_dim = self.vit_model.config.hidden_size
        self.adaptor = nn.Identity()

        if output_type == 'mean':
            self.projection = self.mean_projection
        elif output_type == 'cls':
            self.projection = self.cls_projection
        elif output_type == 'feature':
            self.projection = self.seq_projection
        elif output_type == 'max':
            self.projection = self.max_projection
        elif output_type.startswith('gem'):
            p = float(output_type.split('_')[1])
            self.projection = self.gem_projection
            self.p = nn.Parameter(torch.ones(1) * p)
            if freeze:
                self.p.requires_grad = False
        else:
            raise NotImplementedError(f"Output type {output_type} is not implemented")

    def mean_projection(self, x):
        return x[:, 1:, :].mean(dim=1)
    
    def max_projection(self, x):
        x_bhl = x.permute(0, 2, 1)
        xp_bh = F.adaptive_max_pool1d(x_bhl, output_size=1).squeeze(2)
        return xp_bh
    
    def gem_projection(self, x, eps=1e-6):
        x_clamped = F.relu(x).clamp(min=eps)
        gem_pooled = (x_clamped.pow(self.p).mean(dim=1, keepdim=False)).pow(1./self.p)
        return gem_pooled
    
    def cls_projection(self, x):
        return x[:, 0, :]
    
    def seq_projection(self, x):
        return x

    def forward(self, pixel_values):
        outputs = self.vit_model(pixel_values=pixel_values)
        last_hidden_states = outputs.last_hidden_state
        embs = self.projection(last_hidden_states)
        embs = self.adaptor(embs)
        return embs


class DINOv2Embedder(nn.Module):
    def __init__(self, model_type, output_type='mean', freeze=True, weights="DEFAULT"):
        super().__init__()
        if model_type == "dinov2_vits14":
            self.dino_model = Dinov2Model.from_pretrained("facebook/dinov2-small")
        elif model_type == "dinov2_vitb14":
            # 使用本地模型
            local_path = r'D:\SceneGraph\code\MSG\models--facebook--dinov2-base\snapshots\f9e44c814b77203eaa57a6bdbbd535f21ede1415'
            self.dino_model = Dinov2Model.from_pretrained(local_path,local_files_only=True)

            # self.dino_model = Dinov2Model.from_pretrained("facebook/dinov2-base")
        elif model_type == "dinov2_vitl14": 
            self.dino_model = Dinov2Model.from_pretrained("facebook/dinov2-large")
        else:
            raise NotImplementedError(model_type)

        if freeze:
            for param in self.dino_model.parameters():
                param.requires_grad = False
            self.dino_model.eval()
        
        self.feature_dim = self.dino_model.encoder.layer[-1].mlp.fc2.out_features
        self.adaptor = nn.Identity()

        if output_type == 'mean':
            self.projection = self.mean_projection
        elif output_type == 'cls':
            self.projection = self.cls_projection
        elif output_type == 'feature':
            self.projection = self.seq_projection
        elif output_type == 'max':
            self.projection = self.max_projection
        elif output_type.startswith('gem'):
            p = float(int(output_type.split('_')[1]))
            self.projection = self.gem_projection
            self.p = nn.Parameter(torch.ones(1) * p)
            if freeze:
                self.p.requires_grad = False
        else:
            raise NotImplementedError

    def mean_projection(self, x):
        return x[:, 1:, :].mean(dim=1)
    
    def max_projection(self, x):
        x_bhl = x.permute(0, 2, 1)
        xp_bh = F.adaptive_max_pool1d(x_bhl, output_size=1).squeeze(2)
        return xp_bh
    
    def gem_projection(self, x, eps=1e-6):
        x_clamped = F.relu(x).clamp(min=eps)
        gem_pooled = (x_clamped.pow(self.p).mean(dim=1, keepdim=False)).pow(1./self.p)
        return gem_pooled        
    
    def cls_projection(self, x):
        return x[:, 0, :]
    
    def seq_projection(self, x):
        return x

    def forward(self, pixel_values):
        inputs = {'pixel_values': pixel_values}
        outputs = self.dino_model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        embs = self.projection(last_hidden_states)
        embs = self.adaptor(embs)
        return embs        


Embedders = {
    'resnet50': lambda **kwargs: ResNetEmbedder(model_type='resnet50', **kwargs),
    'resnet18': lambda **kwargs: ResNetEmbedder(model_type='resnet18', **kwargs),
    'convnext-tiny': lambda **kwargs: ConvNextEmbedder(model_type='convnext-tiny-224', **kwargs),
    'convnext-small': lambda **kwargs: ConvNextEmbedder(model_type='convnext-small-224', **kwargs),
    'convnext-base': lambda **kwargs: ConvNextEmbedder(model_type='convnext-base-224', **kwargs),
    'dinov2-small': lambda **kwargs: DINOv2Embedder(model_type="dinov2_vits14", **kwargs),
    'dinov2-base': lambda **kwargs: DINOv2Embedder(model_type="dinov2_vitb14", **kwargs),
    'dinov2-large': lambda **kwargs: DINOv2Embedder(model_type="dinov2_vitl14", **kwargs),
    'vit-base': lambda **kwargs: ViTEmbedder(model_type="vit_base", **kwargs),
    'vit-large': lambda **kwargs: ViTEmbedder(model_type="vit_large", **kwargs),
    'vit-huge': lambda **kwargs: ViTEmbedder(model_type="vit_huge", **kwargs)
}


class Backbone(nn.Module):
    """
    Backbone 网络，用于提取多视角图像的图像特征
    
    输入: 多视角图像 batch
    输出: image-level features (空间特征) 和 global features (全局特征)
    """

    def __init__(self, model_type='convnext-tiny', freeze=True, weights='DEFAULT'):
        super().__init__()
        
        self.obj_embedder = Embedders[model_type](
            freeze=freeze,
            weights=weights,
            output_type='feature'
        )
        
        self.place_embedder = Embedders[model_type](
            freeze=freeze,
            weights=weights,
            output_type='feature'
        )
        
        self.feature_dim = self.obj_embedder.feature_dim

    def forward(self, imgs):
        """
        Args:
            imgs: [B, V, C, H, W], B为batch, V为视角数
            
        Returns:
            img_feats: [B, V, C, H', W'] 空间特征
        """
        B, V = imgs.shape[:2]
        
        imgs_reshaped = imgs.reshape(B * V, *imgs.shape[2:])
        
        img_feats = self.obj_embedder(imgs_reshaped)
        
        img_feats = img_feats.reshape(B, V, *img_feats.shape[1:])
        
        return img_feats
