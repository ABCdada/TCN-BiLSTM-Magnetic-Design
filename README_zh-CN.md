# TCN-BiLSTM 磁性预测与设计网站

这是一个可在个人电脑上部署的磁场强度预测与磁性元件设计网站。项目已经把训练好的 TCN-BiLSTM 模型、后端接口和网页界面整合在一起，下载完整部署包后即可运行。

## 主要功能

- **Design Center**：设置单相全桥工况和环形磁芯尺寸，生成磁通密度序列，预测磁场强度 `H`，并计算页面展示的磁能量/损耗结果。
- **Test Center**：上传单条 `B` 序列，执行长序列冷启动预测，查看预测曲线并导出结果。
- **Model Libraries**：查看支持的材料模型，并设置模型相关参数。
- **Documents**：查看网站使用方法和部署说明。

网页端目前支持四种材料：**3C92、3C95、FEC014 和 T37**。

## 在本机快速部署

推荐直接从 [最新 Release](https://github.com/ABCdada/TCN-BiLSTM-Magnetic-Design/releases/latest) 下载完整部署包。完整包已经包含网站文件和运行所需的模型权重。

运行环境：

- Python 3.10 或更高版本；
- 需要联网安装 Python 依赖，并加载公共 CDN 提供的网页前端库；
- 支持 Windows、Linux 和 macOS，使用 CPU 即可运行。

### Windows

1. 下载并解压 `TCN-BiLSTM-Magnetic-Design-v1.3.3-full-deployment.zip`。
2. 双击 `start_windows.bat`，也可以在命令提示符中运行该文件。
3. 等待首次安装依赖并完成模型校验。
4. 浏览器打开 <http://127.0.0.1:8000>。

### Linux 或 macOS

```bash
unzip TCN-BiLSTM-Magnetic-Design-v1.3.3-full-deployment.zip
cd TCN-BiLSTM-Magnetic-Design-v1.3.3-full
sh start_linux.sh
```

随后在浏览器中打开 <http://127.0.0.1:8000>。

启动脚本会自动创建本地 `.venv`、安装依赖、校验模型文件并启动网站。如需修改监听地址，可在启动前设置 `MAGNET_LISTEN`；例如在可信局域网内可设置为 `0.0.0.0:8000`。

## 主要目录

```text
app.py              Flask 网站与 API
magnetic_model/     模型加载和推理代码
templates/          网站页面
static/             网站样式和图片
weights/            模型权重与校验清单
scripts/            模型权重校验脚本
start_windows.bat   Windows 启动脚本
start_linux.sh      Linux/macOS 启动脚本
```

## 使用提示

- 首次启动需要安装 PyTorch 等依赖，可能需要几分钟；网页也会从公共 CDN 加载 Tailwind CSS 和 Chart.js。
- 请保持 `weights/` 中的模型和 manifest 文件完整，服务器启动时会自动校验。
- Git 源码目录不包含模型二进制文件；需要直接运行时请下载完整部署包。
- Model Libraries 用于查看模型信息和设置参数；模型训练仍需在线下训练环境中完成。
- 开源许可和第三方组件说明见 [LICENSE](LICENSE) 与 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
