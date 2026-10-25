# 第八轮数学说明

最终实施的公式、推导和实验范围统一见 [DESIGN_ZH.md](DESIGN_ZH.md)。数值导数和软件验证见 [TRANSPORT_NUMERICS.md](TRANSPORT_NUMERICS.md)。文献及已有结论的边界见 [LITERATURE.md](LITERATURE.md)。

本包实施EA、Bures、源度量Bures、校正强度/收缩率的全局或条件化元学习，以及保持正则协方差的类别对比旋转。没有加入额外KL损失、支持内LOO搜索或残差分类头的新联合优化。head-only仅是明确列出的参考。
