"""
模块化注册表

用于注册和获取模型组件，实现真正的模块化架构
"""


class ModuleRegistry(object):
    """
    模块注册表

    使用方式:
        @MODULE_REGISTRY.register('my_module')
        class MyModule(nn.Module):
            ...

        module_class = MODULE_REGISTRY.get('my_module')
        module = module_class(**kwargs)
    """

    def __init__(self, name):
        self.name = name
        self._modules = {}

    def register(self, name=None):
        """注册装饰器"""
        def decorator(cls):
            module_name = name or cls.__name__
            if module_name in self._modules:
                raise ValueError(f"Module {module_name} already registered in {self.name}")
            self._modules[module_name] = cls
            return cls
        return decorator

    def get(self, name):
        """获取模块类"""
        if name not in self._modules:
            available = list(self._modules.keys())
            raise KeyError(f"Module {name} not found in {self.name}. Available: {available}")
        return self._modules[name]

    def list_modules(self):
        """列出所有已注册模块"""
        return list(self._modules.keys())

    def has(self, name):
        """检查模块是否已注册"""
        return name in self._modules

    def build(self, name, **kwargs):
        """构建模块实例"""
        module_cls = self.get(name)
        return module_cls(**kwargs)


# 全局注册表实例
FEATURE_EXTRACTOR_REGISTRY = ModuleRegistry('feature_extractor')
QUERY_DECODER_REGISTRY = ModuleRegistry('query_decoder')
EDGE_HEAD_REGISTRY = ModuleRegistry('edge_head')
SCENE_GRAPH_REGISTRY = ModuleRegistry('scene_graph')
LOSS_REGISTRY = ModuleRegistry('loss')
MATCHER_REGISTRY = ModuleRegistry('matcher')


def register_feature_extractor(name=None):
    """注册特征提取器"""
    return FEATURE_EXTRACTOR_REGISTRY.register(name)


def register_query_decoder(name=None):
    """注册查询解码器"""
    return QUERY_DECODER_REGISTRY.register(name)


def register_edge_head(name=None):
    """注册边预测头"""
    return EDGE_HEAD_REGISTRY.register(name)


def register_scene_graph(name=None):
    """注册场景图模块"""
    return SCENE_GRAPH_REGISTRY.register(name)


def register_loss(name=None):
    """注册损失函数"""
    return LOSS_REGISTRY.register(name)


def register_matcher(name=None):
    """注册匹配器"""
    return MATCHER_REGISTRY.register(name)


def build_feature_extractor(name, **kwargs):
    """构建特征提取器"""
    return FEATURE_EXTRACTOR_REGISTRY.build(name, **kwargs)


def build_query_decoder(name, **kwargs):
    """构建查询解码器"""
    return QUERY_DECODER_REGISTRY.build(name, **kwargs)


def build_edge_head(name, **kwargs):
    """构建边预测头"""
    return EDGE_HEAD_REGISTRY.build(name, **kwargs)


def build_scene_graph(name, **kwargs):
    """构建场景图模块"""
    return SCENE_GRAPH_REGISTRY.build(name, **kwargs)


def build_loss(name, **kwargs):
    """构建损失函数"""
    return LOSS_REGISTRY.build(name, **kwargs)


def build_matcher(name, **kwargs):
    """构建匹配器"""
    return MATCHER_REGISTRY.build(name, **kwargs)


# 打印所有注册模块 (用于调试)
def print_registry():
    """打印所有注册表信息"""
    print("=" * 50)
    print("Module Registry Summary")
    print("=" * 50)
    print(f"\nFeature Extractors: {FEATURE_EXTRACTOR_REGISTRY.list_modules()}")
    print(f"\nQuery Decoders: {QUERY_DECODER_REGISTRY.list_modules()}")
    print(f"\nEdge Heads: {EDGE_HEAD_REGISTRY.list_modules()}")
    print(f"\nScene Graphs: {SCENE_GRAPH_REGISTRY.list_modules()}")
    print(f"\nLosses: {LOSS_REGISTRY.list_modules()}")
    print(f"\nMatchers: {MATCHER_REGISTRY.list_modules()}")
    print("=" * 50)
