# TCN-BiLSTM 瞬态磁特性推理

本项目提供论文所述 TCN-BiLSTM 模型的可复现推理实现，用于根据磁通密度及已有运行信息重构磁场强度 `H` 的未知区间。项目包含 Web 界面、JSON API、命令行入口、模型完整性校验、电路窗口接口，以及确定性变流器状态方程与 AI 磁模型相结合的示例。

英文说明见 [README.md](README.md)。

## 1. 项目范围与重要限制

- 模型采用固定的 **1,000 点窗口**和 **16 MHz** 采样率，每个窗口对应 62.5 微秒。
- 标准预测工况为 `100-900`、`500-500` 和 `900-100`；前一个数字是已知 `H` 点数，后一个数字是模型重构点数。
- 模型在一次前向传播中输出窗口内完整的未知区间，不进行逐点自回归。
- `/api/predict-sequence` 通过多个 1,000 点窗口拼接长序列，但不会把双向网络转换成因果、逐时间步模型。单次请求最多处理 50,000 点和 256 个模型窗口。
- Design Center 与 Test Center 仅提供 3C92、3C95、FEC014 和 T37 四种浏览器冷启动模型，且均在首窗采用 `H=0`。Test Center 以 999 点步长共享边界；Design Center 为复现原网站而直接使用页面抽样数据，以不重叠的 1,000 点窗口传递上一窗末端 H。每个窗口内部仍为一次性非自回归预测，但窗口之间采用递归衔接，因此长选区仍可能传播跨窗误差。
- 对外页面中的 Design Center 仅保留单相全桥流程。环形磁芯外半径 `R`、内半径 `r` 和厚度 `d` 均可独立编辑，仅要求尺寸为正且 `r < R`；三相示例仍可通过独立的变流器 API 调用。
- `circuit-demo` 和 `converter-demo` 都要求完整磁通窗口在推理前可用，因此属于块级或窗口级接口，不是逐时间步闭环神经网络联合仿真。
- 直接使用有效截面积 `Ae` 和有效磁路长度 `le` 的换算，假设磁场近似均匀，主要适用于环形磁芯或等效均匀磁路。气隙磁芯、混合材料磁芯、强非均匀磁路或漏磁显著的结构应由分段磁路或场求解器提供相应的局部 `B`、`H`。
- 公开检查点的采样间隔属于学习表示的一部分。重采样数据不能直接使用；应显式处理 `dB/dt`，并重新训练或微调模型。
- 训练数据覆盖范围之外的温度、磁通密度、直流偏置、饱和状态或激励形式均属于外推结果。

## 2. 材料模型与输入通道

| ID | 材料牌号 | 可训练参数量 | 网络配置 |
|---|---|---:|---|
| A | 3C92 | 497,793 | 5 层、64 通道 TCN；双层 BiLSTM，每方向 96 单元 |
| B | 3C95 | 2,092,545 | 7 层、128 通道 TCN；双层 BiLSTM，每方向 192 单元 |
| C | FEC007 | 2,092,545 | 与 B 相同；科学接口采用 `1-999` 零占位协议 |
| D | FEC014 | 2,512,593 | 7 层、144 通道 TCN；双层 BiLSTM，每方向 208 单元 |
| E | T37 | 2,092,865 | 7 层、128 通道 TCN；双层 BiLSTM，每方向 192 单元；六通道输入 |

上表是完整的五材料**科学模型注册表**，仍保留 C/FEC007 及其科学检查点。浏览器版本采用独立的冷启动注册表：Design Center、Test Center 和 Model Libraries 只显示 3C92、3C95、FEC014、T37 四个牌号，不在页面上显示内部 A/B/D/E 编号。C/FEC007 权重不会被删除，但不注册到浏览器冷启动版本中。

A–D 的五个输入通道为：

1. 完整 `B` 序列；
2. 一阶离散磁通增量 `dB`；
3. 温度 `T`；
4. 已知区间的 `H`；
5. 区分已知区间与待预测区间的掩码。

最终论文所用的 E 材料检查点还包含第二离散磁通增量 `ddB`，因此输入为六通道：`B`、`dB`、`ddB`、`T`、`H_known` 和 `mask`。`dB`、`ddB` 与掩码均由后端自动生成，用户不应自行重复添加。

Material C 没有有效历史 `H` 轨迹，只支持 `1-999` 工况，并要求一个原始零占位值：

```json
{"material": "C", "mode": "1-999", "known_H": [0.0]}
```

## 3. 安装

项目的重要文档与结果入口：

- [模型卡](MODEL_CARD.md)、[数据卡](DATA_CARD.md)与[复现指南](REPRODUCIBILITY.md)；
- [结果目录](results/README.md)，其中分别保存论文汇总指标、独立 PFC 实测验证、CPU/GPU 软件校验和 16 组完整的合成请求/响应示例；
- [参考项目对应关系](docs/REFERENCE_PARITY.md)、原创[模型结构图](docs/images/model_architecture.svg)和[电路工作流程图](docs/images/circuit_workflow.svg)；
- [引用信息](CITATION.cff)与[训练代码发布范围说明](TRAINING.md)。
- [部署指南](DEPLOYMENT.md)、[API 说明](API.md)、[测试指南](TESTING.md)、
  [发布说明](RELEASE_NOTES.md)与[GitHub 上传检查表](GITHUB_UPLOAD_CHECKLIST.md)。

本次发布提供两套包：**GitHub 源码包**包含代码、文档、测试、示例、结果和
manifest，但不包含 `.pth`；**完整部署包**在相同目录结构中包含经两份
manifest 校验的 5 个科学模型和 4 个浏览器冷启动模型。另行生成的科学权重包
和冷启动权重包可在确认再发布权利后作为 GitHub Release 附件。权重边界见
[MODEL_WEIGHTS_NOTICE.md](MODEL_WEIGHTS_NOTICE.md)。

当前页面通过公共 CDN 加载 Tailwind CSS 与 Chart.js。后端推理可在本机运行，
但若未将这些第三方资源改为本地托管，完整的浏览器样式和图表仍需要联网。

建议使用全新的虚拟环境。项目已在 Windows/Python 3.13/PyTorch 2.7 CPU，以及 AutoDL/Python 3.12/PyTorch 2.8/CUDA 12.8 环境中验证。

### Windows PowerShell

```powershell
cd C:\path\to\tcn_bilstm_magnetic_circuit
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Linux/macOS CPU

```bash
cd /path/to/tcn_bilstm_magnetic_circuit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### NVIDIA GPU

先根据驱动版本从 PyTorch 官方渠道安装匹配的 CUDA 版本，再安装其余依赖。例如 CUDA 12.8：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

CPU 与 CUDA 内核可能产生极小的浮点差异。正式报告结果时应保存 Python、PyTorch、CUDA、GPU 型号及 `pip freeze` 信息。

## 4. 模型权重与权威 manifest

将发布方提供的五个原始检查点及其**权威** `manifest.json` 放在同一目录：

```text
weights/
|-- A_best_model.pth
|-- B_best_model.pth
|-- C_best_model.pth
|-- D_best_model.pth
|-- E_best_model.pth
`-- manifest.json
```

启动服务前必须验证文件大小和 SHA-256：

```bash
python scripts/verify_weight_manifest.py --manifest weights/manifest.json
python scripts/verify_all_models.py \
  --manifest weights/manifest.json \
  --device cpu \
  --output outputs/model_verification.json
```

服务加载检查点时也会按照 manifest 验证材料、参数量、文件名、文件大小和 SHA-256，并严格加载网络状态。

> `scripts/build_weight_manifest.py` 只供发布维护者建立正式版本清单。普通部署时不要重新生成 manifest，否则会用当前目录中任意文件的哈希覆盖权威哈希，失去校验意义。

可通过 `MAGNET_MODEL_DIR` 指定权重目录，或使用 `MAGNET_CHECKPOINT_A` 至 `MAGNET_CHECKPOINT_E` 覆盖单个检查点。进行对比实验时必须记录这些设置。

## 5. 启动 Web 服务

本地开发：

```bash
flask --app app run --host 127.0.0.1 --port 8000
```

本地或远程正式运行：

```bash
waitress-serve --listen=0.0.0.0:8000 app:app
```

浏览器打开 <http://127.0.0.1:8000>。服务状态和模型信息接口为：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/api/models
```

Web 页面提供单窗口预测、数据上传、`B-H` 轨迹显示，以及确定性变流器窗口示例。Design Center 与 Test Center 使用 3C92、3C95、FEC014、T37 四种独立冷启动模型；每个模型的首窗都采用训练时的单零锚点 `1-999` 协议。科学计算 API 仍保留原有 A–E 五种材料（包括 C/FEC007）及其历史 `H` 输入约定。Test Center 按 16 MHz 输入并以 999 点步长递归；为尽可能复现原网站数值，Design Center 直接使用页面每个开关周期保存的 10 个点，不插值到 16 MHz，并采用不重叠的 1000 点窗口、在窗口边界传递上一窗末端 H。网页会完整处理所选时间区间。所有接口错误均采用统一结构：

科学兼容接口 `/api/predict` 的 `materialId` 仍为 1–5，依次对应 3C92、3C95、FEC007、FEC014、T37。Design Center 与 Test Center 调用的是独立的 `/api/predict-cold-start`；其浏览器 `materialId` 仅为 1–4，依次对应 3C92、3C95、FEC014、T37。页面只显示牌号名称，A/B/D/E 仅作为内部注册标识。Model Libraries 同样只列出这四个牌号；基线和超参数输入项可以编辑，所选配置保存在浏览器本地，但文件上传、服务器训练和模型/报告导出在本推理版本中仍保持离线，不会因修改输入框而启动训练。

```json
{"error": {"code": "invalid_request", "message": "...", "details": {}}}
```

## 6. 推理 API

### 6.1 `POST /api/predict-window`

重构一个 1,000 点窗口：

```json
{
  "material": "A",
  "B": ["1000个有限数值，单位T"],
  "temperature": 25.0,
  "known_H": ["已知前缀，单位A/m"],
  "mode": "100-900",
  "sampling_rate_hz": 16000000
}
```

`temperature` 可以是标量，也可以是长度为 1,000 的序列。A、B、D、E 必须提供与工况相符的已知 `H` 前缀；C 使用前述单个零占位值。响应中的 `H` 是完整重构序列，`predicted_H` 仅包含目标区间。

可先生成确定性示例请求：

```bash
python scripts/generate_sample_inputs.py --output-dir examples/generated
curl -X POST http://127.0.0.1:8000/api/predict-window \
  -H "Content-Type: application/json" \
  --data-binary @examples/generated/predict_window_A.json
```

### 6.2 `POST /api/predict-sequence`

对不少于 1,000 点的长序列进行窗口拼接。核心字段为 `material`、`B`、`temperature`、`known_H`、`history_length`，并可选提供 `hop_length`。

A、B、D、E 的后续窗口只使用已经重构的 `H` 作为历史前缀；Material C 则在每个窗口继续使用其训练时定义的单个原始零占位值。重叠目标区间采用平均融合。此功能是窗口级拼接，不代表模型对任意长序列具有因果性，也不代表长序列精度已自动得到验证。

上述科学计算接口与 Design Center 的末端 `H` 递推兼容流程彼此独立，前者仍保持论文评测时的材料 C 逐窗零占位语义。

### 6.3 `POST /api/circuit-demo`

该接口用于把外部电路求解器或实测数据产生的感应电压窗口接入磁模型：

```json
{
  "material": "A",
  "induced_voltage": ["磁芯感应电压序列"],
  "current": ["已知区间的绕组电流"],
  "temperature": 25.0,
  "turns": 20,
  "area_m2": 0.0001,
  "path_length_m": 0.08,
  "sampling_rate_hz": 16000000,
  "initial_B": 0.0,
  "history_length": 100
}
```

这里的 `induced_voltage` 必须是磁芯感应电压，而不是未经处理的端电压。实际系统中应先扣除绕组电阻、漏感和相关寄生元件的压降，再根据法拉第定律积分得到 `B`。

已知状态可用与历史长度相同的 `current`，也可直接用 `known_H` 提供；未知目标区间不需要电流真值。若要计算独立验证误差，可额外提供完整的 `reference_current`。没有完整参考时，验证指标返回 `null` 及原因说明。

若需要复现论文或比赛的能量误差分母，还应提供单位为 J/m³ 的正数 `reference_loss_density`，其含义是实测完整周期损耗能量密度。

### 6.4 `POST /api/converter-demo`

该接口先求解一个确定性的 1,000 点变流器窗口，再把感应电压和名义电流历史交给 AI 磁模型：

```json
{
  "material": "A",
  "mode": "100-900",
  "topology": "single_phase_full_bridge",
  "temperature": 25.0,
  "turns": 20,
  "area_m2": 0.0001,
  "path_length_m": 0.08,
  "sampling_rate_hz": 16000000,
  "initial_B": 0.0,
  "dc_voltage": 800.0,
  "grid_voltage_rms": 220.0,
  "grid_frequency_hz": 50.0,
  "switching_frequency_hz": 100000.0,
  "filter_inductance_h": 0.000145,
  "series_resistance_ohm": 0.2,
  "initial_current_A": 0.5,
  "grid_phase_deg": 0.0
}
```

支持的拓扑为：

- `single_phase_full_bridge`：单相全桥；
- `three_phase_full_bridge`：三相两电平全桥；
- `three_phase_t_type`：三相 T 型三电平变流器。

电气前端采用确定性正弦 PWM 和串联 `R-L` 状态方程，使用显式 Euler 法更新名义电流，并计算逆变器电压、电网电压和电感感应电压。随后，后端用名义电流的已知前缀初始化磁模型，并在完整 `B` 窗口上执行一次 AI 推理。

该示例用于展示“**确定性变流器状态方程 + AI 窗口后处理**”的集成流程。AI 预测的 `H` 不会反馈到每个电路时间步，名义 `R-L` 电流也不是实测参考，因此它不是闭环逐时步神经网络共仿真，也不能作为实验精度验证。若需要强耦合仿真，外部求解器必须实现块级迭代、状态更新和收敛判据。

## 7. 感应电压与磁路几何要求

在均匀环形磁路近似下，接口使用以下关系：

- 对磁芯感应电压积分，并结合匝数与有效截面积得到 `B`；
- 根据绕组电流、匝数与有效磁路长度得到 `H`；
- 根据预测的 `H`、有效磁路长度与匝数得到磁化支路电流。

这些换算不能直接代表气隙附近、多个材料拼接处或强漏磁区域的局部场量。对于混合磁性结构，应由分段磁路模型或有限元场求解器生成局部磁通密度和场强，再按模型的采样及输入约定调用推理接口。

## 8. 能量量的解释

项目采用梯形积分计算离散路径上的 `H dB`，必须区分以下概念：

1. **开放瞬态区间积分**：表示该区间内单位体积的净磁能交换，其中可能包含可逆储能变化，不能直接称为不可逆磁芯损耗。
2. **闭合路径积分**：表示所提供完整闭合路径的累计能量。只有路径恰好包含一个磁周期时，才可解释为每周期损耗能量密度；多个周期的记录得到的是累计值。
3. **磁芯损耗功率密度**：单周期损耗能量密度乘以实际周期频率，单位为 W/m³。只有在积分区间与相应闭合周期一致时，才能直接使用开关频率换算。

## 9. 测试与复现

生成不含私有数据的示例输入并运行测试：

```bash
python scripts/generate_sample_inputs.py --output-dir examples/generated
pytest -q
```

生成包含请求 JSON、响应 JSON、逐点 CSV、汇总表和可编辑 SVG 的完整示例包：

```bash
python scripts/generate_reproducible_bundle.py \
  --project-root . \
  --output results/reproducible_examples \
  --device cpu
```

该示例包只用于软件链路校验，全部明确标注为合成数据，不含实测目标 `H`，因此不能用于声明预测精度。

安装真实检查点后，运行完整模型验证：

```bash
python scripts/verify_weight_manifest.py --manifest weights/manifest.json
python scripts/verify_all_models.py \
  --manifest weights/manifest.json \
  --device cpu \
  --output outputs/model_verification.json
```

这些测试验证模型注册表、API 输入校验、检查点完整性、网络参数量、前向形状、窗口拼接、电路接口、变流器状态方程和 `H dB` 数值积分。它们验证的是软件与物理量接口，不等同于复现论文精度；精度复现仍需要原始评测记录、数据划分和对应真值。

发布结果时建议保存：

- Git 提交号及权威 `weights/manifest.json`；
- Python、PyTorch、CUDA 和 GPU 信息；
- `pip freeze`；
- 材料、采样率、温度、历史长度和窗口步长；
- 指标采用开放区间、单周期闭合路径还是多周期累计路径；
- 在生成重叠窗口之前完成的采集序列级训练、验证和测试划分。

## 10. AutoDL 运行示例

以下命令仅使用占位符，不应将真实密码、Token 或 SSH 凭证写入仓库：

```bash
ssh -p <PORT> root@<AUTODL_HOST>
cd /path/to/persistent-data
git clone https://github.com/ABCdada/TCN-BiLSTM-Magnetic-Design.git tcn_bilstm_magnetic_circuit
cd tcn_bilstm_magnetic_circuit

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

mkdir -p weights
# 将 A_best_model.pth ... E_best_model.pth 与发布方提供的
# 权威 manifest.json 一起复制到 weights/，不要自行重建 manifest。
python scripts/verify_weight_manifest.py --manifest weights/manifest.json
python scripts/verify_all_models.py --device cpu --output outputs/model_verification.json
pytest -q

waitress-serve --listen=0.0.0.0:8000 app:app
```

通过 AutoDL 的自定义服务或端口映射功能访问 8000 端口，优先使用平台分配的 HTTPS 地址，并在公开部署时配置身份认证、访问限制与反向代理超时。

## 11. Docker

默认镜像使用 CPU 版 PyTorch，权重不会写入镜像，应以只读卷挂载：

```bash
docker build -t tcn-bilstm-magnetic:cpu .
docker run --rm -p 8000:8000 \
  -v "$PWD/weights:/app/weights:ro" \
  tcn-bilstm-magnetic:cpu
```

Windows PowerShell 的卷参数可写为 `${PWD}/weights:/app/weights:ro`。

## 12. 开源关系与许可

本版本在 MIT 许可下采用了 [LSTM-DAO-Inference](https://github.com/ouihon/LSTM-DAO-Inference) 的网站布局，并将预测后端、五材料科学模型注册表、四牌号浏览器冷启动注册表、归一化、验证代码和结果文件替换为本项目的 TCN-BiLSTM 实现，同时采用独立的 Aurora 配色。逐项文件与功能对应关系见 [docs/REFERENCE_PARITY.md](docs/REFERENCE_PARITY.md)，上游完整许可文本见 [licenses/LSTM-DAO-Inference-LICENSE.txt](licenses/LSTM-DAO-Inference-LICENSE.txt)。

本项目原创代码和文档采用 MIT License。检查点和数据集可能具有独立使用条款，详见 [LICENSE](LICENSE) 和 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
