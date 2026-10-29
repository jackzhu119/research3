# 正式科研图件

本目录包含4张主图与4张补充图。全部由真实归档预测或核验后的数值表使用Matplotlib确定性绘制；计算值和置信区间与独立分析输出逐项对应。

## 投稿与编辑格式

- TIFF：600dpi、RGB、LZW压缩，可单独上传期刊。
- PNG：600dpi、RGB，便于预览与插入稿件。
- PDF：矢量版本，适合正式排版。
- SVG：文字可编辑的矢量版本，可用Illustrator或Inkscape继续编辑。

图件使用统一DejaVu Sans无衬线字体、明确的A/B/C/D面板标记、坐标轴单位、参与者人数和图注。纠错C统一绿色，引错I统一橙色，净变化G统一黑色；置信区间与参与者配对关系按各图定义显示。各队列保持自己的参考模型、预算轴和单位。

`montage_review.jpg`是八图的总览预览。`captions.md`提供完整英文图注；`proof.json`和`STYLE_REVIEW.json`（如提供）记录数值和版式检查；`input/`保留绘图输入。运行 `python figures/plot_integrated.py` 可从附带CSV再生成全部格式。数值分析和数据来源说明见完整论文的数据代码包。
