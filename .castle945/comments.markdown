## 核心组件
<!-- python train.py = Runner.from_cfg/Runner.__init__ + Runner.train -->
- Runner
	- __init__
		"""执行器初始化
		- 惰性初始化（数据集和优化器封装）即先只拷贝配置参数稍后在任务循环中实例化
		- 设置多卡环境和随机种子
		- 创建日志器、消息枢纽、可视化器
		- 创建模型并在其中创建数据预处理器，模型中各模块类构造函数的调用关系和其在配置文件的嵌套一样
		- 注册钩子，即按照配置文件中的顺序调用各钩子类的构造函数
		Notes:
			* 默认钩子的逻辑：默认自带 6 个钩子，每个默认钩子都有都用默认参数，如果传入某个默认钩子的参数为 None 则弹出禁用该钩子，不为 None 则更新该默认钩子的参数
		"""
		```
		- setup_env/init_dist + set_randomness/set_random_seed
		- build_logger + build_message_hub + build_visualizer/[VISUALIZERS]/[VISBACKENDS]
		- build_model/[MODELS]/[DATAPREPROC]
		- register_hooks/(*register_default_hooks* + register_custom_hooks)/[HOOKS]
		```
	- train
		"""执行器执行训练任务
		- 创建训练循环/训练数据集/训练数据变换
		- 创建优化器封装 + 执行学习率自动缩放 + 创建参数调度器
		- 创建验证循环/(验证数据集/验证数据变换 + 评测器/评价指标)
		- 执行 before_run 位点钩子，即依次调用各钩子的 before_run 方法，其他位点同理
		- 加载预训练模型或恢复训练 + 执行训练循环
		- 执行 after_run 位点钩子
		Notes:
			* 学习率自动缩放：例如基准配置在 8xb6（即 8 卡每卡批量 6 ）时使用配置文件中的基准学习率，当实际运行配置为 2xb8 时，学习率自动缩放根据实际总批量/基准总批量 =16/48 的比率来缩放基准学习率
			* default_collate 和 pseudo_collate：前者会合并列表并增加批量维度，后者保持列表结构不变，MMEngine 默认后者，但是 Pytorch 的默认行为是前者
		"""
		```
		- build_train_loop/[LOOPS(EpochBasedTrainLoop|IterBasedTrainLoop)]
			- BaseLoop/*Runner.build_dataloader*/[DATASETS]/BaseDataset/[TRANSFORMS]
		- build_optim_wrapper/[OPTIM_WRAPPERS] + *scale_lr* + build_param_scheduler/[PARAM_SCHEDULERS]
		- build_val_loop/[LOOPS(ValLoop)]
			- BaseLoop/Runner.build_dataloader/[DATASETS]/BaseDataset/[TRANSFORMS]
			- [EVALUATOR]/[METRICS]
		- Runner.call_hook('before_run')/[HOOKS].before_run
		- load_or_resume + **[LOOPS(EpochBasedTrainLoop|IterBasedTrainLoop)].run**
		- Runner.call_hook('after_run')/[HOOKS].after_run
		```
	- test
		"""执行器执行测试任务，与验证过程一模一样
		- 创建测试循环/(测试数据集/测试数据变换 + 评测器/评价指标)
		- 执行 before_run 位点钩子，即依次调用各钩子的 before_run 方法，其他位点同理
		- 加载预训练模型或恢复训练 + 执行测试循环
		- 执行 after_run 位点钩子
		"""
		```
		- build_test_loop/[LOOPS(TestLoop)]
			- BaseLoop/Runner.build_dataloader/[DATASETS]/BaseDataset/[TRANSFORMS]
			- [EVALUATOR]/[METRICS]
		- Runner.call_hook('before_run')/[HOOKS].before_run
		- load_or_resume + **[LOOPS(TestLoop)].run**
		- Runner.call_hook('after_run')/[HOOKS].after_run
		```
- EpochBasedTrainLoop
	- run
		"""执行基于轮数的训练循环
		- 执行各个位点钩子
		- 当训练轮数未到时执行一轮迭代（迭代数据集长度次数为一轮迭代）
		- 每次迭代中先迭代训练数据集获得一个批量数据， 再执行模型训练步骤
		- 每轮次结束时根据当前已训练轮数决定是否执行验证循环
		Notes:
			* 基于轮数的训练循环：① 执行各个位点钩子 ② 当训练轮数未到时执行一轮迭代（迭代数据集长度次数为一轮迭代）③ 每次迭代中先迭代训练数据集获得一个批量数据， 再执行模型训练步骤 ④ 每轮次结束时根据当前已训练轮数决定是否执行验证循环
		"""
		```
		- Runner.call_hook('before_train')/[HOOKS].before_train
		- EpochBasedTrainLoop.run_epoch
			- Runner.call_hook('before_train_epoch')/[HOOKS].before_train_epoch
			- **[DATASETS].getitem** + pseudo_collate
			- EpochBasedTrainLoop.run_iter
				- Runner.call_hook('before_train_iter')/[HOOKS].before_train_iter
				- **[MODELS].train_step**
				- Runner.call_hook('after_train_iter')/[HOOKS].after_train_iter
			- Runner.call_hook('after_train_epoch')/[HOOKS].after_train_epoch
		- **[LOOPS(ValLoop)].run**
		- Runner.call_hook('after_train')/[HOOKS].after_train
		```
- IterBasedTrainLoop
	"""执行基于迭代次数的训练循环
	- 执行各个位点钩子
	- 当迭代次数未到时执行一次迭代（数据集转可迭代类型可无限迭代）
	- 每次迭代中先迭代训练数据集获得一个批量数据，再执行模型训练步骤
	- 每迭代结束时根据当前已迭代次数决定是否执行验证循环
	Notes:
		* 基于迭代次数的训练循环：① 执行各个位点钩子 ② 当迭代次数未到时执行一次迭代（数据集转可迭代类型可无限迭代）③ 每次迭代中先迭代训练数据集获得一个批量数据，再执行模型训练步骤  ④ 每迭代结束时根据当前已迭代次数决定是否执行验证循环
	"""
	```
	- Runner.call_hook('before_train')/[HOOKS].before_train
	- Runner.call_hook('before_train_epoch')/[HOOKS].before_train_epoch
	- **[DATASETS].getitem** + pseudo_collate
	- IterBasedTrainLoop.run_iter
		- Runner.call_hook('before_train_iter')/[HOOKS].before_train_iter
		- **[MODELS].train_step**
		- Runner.call_hook('after_train_iter')/[HOOKS].after_train_iter
	- **[LOOPS(ValLoop)].run**
	- Runner.call_hook('after_train_epoch')/[HOOKS].after_train_epoch (注: 此类训练循环中此位点应为空)
	- Runner.call_hook('after_train')/[HOOKS].after_train
	```
- ValLoop
	"""执行验证循环
	- 执行各个位点钩子
	- 每次迭代中先迭代验证数据集获得一个批量数据，再执行模型验证步骤，再评测器处理模型输出
	- 所有迭代结束后计算评价指标
	Funcs:
		_update_losses: 输入模型预测结果和当前损失字典，用模型预测结果中的损失字典更新当前损失字典
		_parse_losses: 输入当前损失字典，计算每项损失的均值并返回一个新字典其键为损失名值为损失的均值
	Notes:
		* 验证循环：① 执行各个位点钩子 ② 每次迭代中先迭代验证数据集获得一个批量数据，再执行模型验证步骤，再评测器处理模型输出 ③ 所有迭代结束后计算评价指标
	"""
	```
	- Runner.call_hook('before_val')/[HOOKS].before_val
	- Runner.call_hook('before_val_epoch')/[HOOKS].before_val_epoch
	- **[DATASETS].getitem** + pseudo_collate
	- ValLoop.run_iter
		- Runner.call_hook('before_val_iter')/[HOOKS].before_val_iter
		- **[MODELS].val_step**
		- _update_losses + **[EVALUATOR].process**
		- Runner.call_hook('after_val_iter')/[HOOKS].after_val_iter
	- **[EVALUATOR].evaluate**
	- _parse_losses + [METRICS].update
	- Runner.call_hook('after_val_epoch')/[HOOKS].after_val_epoch
	- Runner.call_hook('after_val')/[HOOKS].after_val
	```
- TestLoop <!-- 略，与验证循环一模一样 -->
## 模块实现层
## 数据集
- BaseDataset
	"""数据集基类
	- 初始化
		- 加载数据集元信息、拼接根目录、创建数据变换管道
		- 读文件加载数据信息列表并处理
		- 过滤非法数据、数据信息列表序列化为字节流以节省内存
	- 获取样本
		- 获取当前样本的数据信息并执行数据变换管道
	Funcs:
		parse_data_info: 默认只做路径拼接，子类中重写以实现额外计算变换矩阵等处理
		filter_data: 默认不过滤，子类中重写以实现如过滤在训练集中却无标签的数据（ONCE 数据集中存在）
		prepare_data: 默认只做获取当前样本的数据信息并执行数据变换管道，子类中重写以实现额外功能
	Notes:
		* 数据集元信息的优先级：有多种来源，取高优先级，数据集构造函数形参中传入的元信息 > 数据集类属性中的元信息 > 数据标注文件中的元信息
		* 数据文件的路径处理：_join_prefix 中将 data_root（根目录）拼接上 data_prefix（数据文件的前缀目录）得到相对目录，再在 parse_data_info 中和 data_info（数据信息）中的数据文件名拼接得到完整路径
	"""
	```
	- __init__
		- **_load_metainfo** + *_join_prefix* + Compose/[TRANSFORMS]
		- full_init
			- load_data_list/**parse_data_info**
			- *filter_data* + _serialize_data
	- __getitem__
		- prepare_data/(~~get_data_info~~ + [TRANSFORMS].transform)
	```
- pseudo_collate
	"""不同于 PyTorch 默认的 default_collate 会直接将样本堆叠成张量，此'伪'批处理函数保持字典的结构不变并深入字典元素将字典中的列表或元组按位置合并
	- 若批量中的数据样本（以第一个样本判定）类型为字符串或字节，无法转张量返回原值
	- 若是具名元组，将同名字段合并为元组，过程为: 批量是具名元组的列表，*lst 解包作为位置参数，zip(*lst) 即 zip(item0, item1,...)，(for ) 是元组生成式得到 ((name1,name2,..), (age1, age2,...),...) 最后解包作为位置参数用来构造具名元组(可以视为类类型)，得到 [Person(name=name1,...), Person(name=name2,...)] -> Person(name=(name1,...), age=(age1,...),...)
	- 若是列表或元组，先检查所有样本中该列表的长度均相等，再将某个位置处来自所有样本的该处的元素合并成元组，即 [[a1,b1], [a2,b2],...] -> [[(a1,a2,...), (b1,b2,...)]]
	- 若为字典，则展开字典递归并按键合并即 [{0:a1,...}, {0:a2,...},...] -> [{0:[a1,a2,...],...}]
	"""
#### 数据预处理器
- BaseDataPreprocessor
	"""数据预处理器基类，基类只遍历批量数据并将数据搬到目标设备，具体来说 str 不变，Tensor 直接搬到显存，列表或字典等类型遍历逐个搬运"""
- ImgDataPreprocessor
	"""图像数据预处理器，额外支持图像归一化、边界填充等低计算量数据变换，相比在数据变换中执行，数据预处理器中执行可以提升数据搬运效率，因为前者处理后搬运的是 float32 数据而后者可以先搬运 uint8 再处理
	- 递归搬运数据到目标设备
	- 如果输入数据为图像列表，逐图像执行 交换颜色通道、图像归一化、边界填充
	- 如果输入数据为图像张量，对此图像张量（可能包含多张图像）的依次执行相同的数据变换
	Args:
		mean/std: 图像归一化参数
		pad_size_divisor/pad_value: 边界填充参数
		bgr_to_rgb/rgb_to_bgr: 任一项设置则执行交换颜色通道
	"""
#### 模型
- BaseModule
	- **init_weights**
		"""
		- 标记顶层调用者，创建权重初始化信息字典并挂载到所有子模块
		- 若 init_cfg 非空，调用 initialize 执行基于配置字典的初始化
		- 深度优先遍历子模块并执行其自定义初始化 init_weights
		- 预训练权重最后加载
		Locals:
			is_top_level_module: 标记当前 self 是否为顶层调用者，即 model，假设调用链为 model.init_weights() -> model.backbone.init_weights() -> model.backbone.conv1.init_weights()，不需要传入，顶层调用后会为子模块添加 _params_init_info 属性，有此属性不视为顶层调用者
			_params_init_info (dict): 权重初始化信息字典，键为 nn.Parameter 类型的权重 (比如 conv1.weight) 的引用，值为该权重的初始化信息字典 ('init_info' 记录 "我最近被哪个模块初始化" 的字符串，'tmp_mean_value' 记录当前权重的均值作为指纹用于判断权重是否被更改)，注意此字典全模型共享 (顶层和子模块都有此属性)
			_is_init: 标记当前模块是否已初始化，防重复，例如当误调用多次 model.init_weights() 或 当前模块为共享模块 (例如分支 self.b1, self.b2 = nn.Sequential(shared_encoder, nn.Conv2d), ... 中的 shared_encoder)
		Notes:
			* 关于自定义变量初始化权重参数：init_weights 在 BaseModule 中定义而非 Pytorch 接口，在 `Runner.train/Runner._init_model_weights` 中调用，而 `model.cuda()` 的操作在 `Runner.__init__/Runner.wrap_model` 中执行，故用自定义变量初始化权重参数需考虑设备类型
			* 关于 init_cfg 和 init_weights：见 `BaseModule.init_weights`，两者都会执行，先执行 init_cfg 再执行 init_weights，init_cfg 适合批量初始化如 Linear 类的初始化方式，init_weights 自定义初始化可以针对具体某个变量做
		"""
- BaseModel
	"""模型基类
	- 初始化
		- 创建数据预处理器
	- 训练步骤
		- 执行优化器封装的上下文
		- 执行数据预处理器，包括加载数据到显存、批量数据变换等
		- 执行模型前向
		- 解析损失字典 + 优化器封装执行反向传播更新模型参数
	- 验证/测试步骤
		- 执行数据预处理器，包括加载数据到显存、批量数据变换等
		- 执行模型前向
	Funcs:
		_run_forward: 批量数据解包作为 forward 的形参并调用 forward，一般不改而是让 forward 包含 kwargs 来装不用的参数
		parse_losses: 解析损失字典，返回可以 backward 的损失以及可以被日志记录的损失
	"""
	```
	- __init__/[DATAPREPROC]
	- ***train_step***
		- [OPTIM_WRAPPERS].optim_context
		- [DATAPREPROC].forward
		- _run_forward/[MODELS].forward
		- parse_losses + [OPTIM_WRAPPERS].update_params
	- (val_step|test_step)
		- [DATAPREPROC].forward
		- _run_forward/[MODELS].forward
	```
#### 可视化器和可视化后端
- Visualizer (ManagerMixin)
	"""可视化器，包含两部分内容，一是封装 cv2 等的绘图绘点线框等（此部分无用），二是调用可视化后端保存运行时信息如 add_scalar 等
	- 初始化
		- 遍历可视化后端列表逐个创建可视化后端
		- 若用 matplotlib 绘图时，用传入的形参初始化画布参数
		- 若传入图像，将图像放到 matplotlib 的画布上
	- 可视化显示，初始化窗口并显示
	Args:
		image: 待可视化的数据，RGB 图像
		vis_backends (list): 可视化后端配置列表
		save_dir: 可视化后端需要的数据保存目录
		fig_save_cfg/fig_show_cfg: 用 matplotlib 保存图或绘图时需要的画布参数
	Funcs:
		draw_bboxes|draw_lines|draw_texts|draw_circles|draw_points 等: 在 matplotlib 画布上画框、线、文本等
		add_configs|add_scaler|add_datasample 等: 逐个可视化后端调用相应的方法保存配置、标量信息、数据样本等
		set_image|get_image: 图像上画布或从画布获取图像
	"""
- Hook
	"""接口类定义 before_run 等方法"""
