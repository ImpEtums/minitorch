## 配置环境
官方文档中提到 “MiniTorch requires Python 3.11 or higher.”，实际使用发现，需要创建 python==3.9 的虚拟环境，否则会出现许多依赖冲突。
由于 ssl 问题，无法直接使用 pip 安装 requirements.txt 中的包（特别是Colorama），使用了conda安装来曲线救国。
在官方的教程中，没有提到需要执行 conda install llvmlite ，但查阅网上相关贴文发现可能是需要安装的。
可视化需要用到的 altair 是v4，但默认安装的是v5，需要手动降级到4.2.2
