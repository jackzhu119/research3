# Brain Sciences 英文研究稿与可复现图表

**稿件标题：Evidence-Retaining Functional Calibration for Few-Trial Motor Imagery EEG Decoding**

这是供三位作者审阅、修改和确认声明的完整英文初稿，不是已提交或已录用文件。主稿按期刊允许的free-format独立排版；没有复制MDPI官方模板用于网络发布。

## 文件入口

- Manuscript.docx：可编辑英文主稿，含四段结构摘要、3张可编辑表、4组主图、可编辑公式和21条数字引用。
- Manuscript.pdf：主稿排版预览。
- Supplementary_Materials.docx / .pdf：补充方法与表S1–S6、历史保护机制图S1。
- submission/Cover_Letter.docx：投稿信草稿，排他投稿/全体批准等事实须确认后替换显式待填段。
- submission/Submission_Check_ZH.md：最新投稿政策、伦理依据与待作者确认项目。
- figures/：Matplotlib实际数据绘图；可编辑文字SVG、矢量PDF、600dpi TIFF/PNG，另附图注和绘图代码。
- tables/：全部主比较、18项事后机制检验、配对参与者数据与图源CSV。
- evidence/：准确结果、源清单SHA、实际模型/协议/种子与原始来源链接。
- research/：六篇截图文献的书目信息、合法读取范围、结构笔记、期刊规则与中国伦理条款核查。

## 科研主线与原创边界

ERFC是本项目已实际运行的query-functional profile校准规则的论文命名，对应原profile-v1/a0.3-c0.1，并非新增训练或新发明EEGNet。贡献落在支持证据增益保留集合、允许支持之外的函数完成、以及逐查询最小改动决策这一明确组合机制。既有EEGNet、EA、核与Schur等基础保留正确出处。

主稿使用第10轮已完成的真实数据，35名数据集参与者、143会话、429会话–种子单元；统计单位是参与者。12个原始准确率主要比较未形成校正后的优越性结论，论文没有写成SOTA、等效、临床安全保证或独立验证。新增18项探索性机制检验显示少引入错误，也少纠正原错误，二者同表同图呈现。

第3轮4.66个百分点的保护效果只用于明确标注的历史机制补充，不挪作ERFC主结果。第12/13轮同源pilot不冒充独立新队列。已终止的第1/11轮没有被当作完整新确认性实验。

## 数据与稿件版本

核查时仓库有13个实验结果分区、46个Git分支；协议v2.0、信号输入合同v2.1，W0/W1是执行阶段。已检查范围未发现既有正式投稿稿件。当前文件为本次基于固定结果快照创建的英文稿D1，不把实验轮次或管理分支算作多个正式稿。

源结果固定版本：a16d06760ae77699a809c9ba36d002d684560c82。

## 文献读取范围

六篇截图文献已逐项核准确题名、作者、年份和DOI。Stiefel-SPD取得合法机构全文并完成阅读；其余五篇未找到合法公开全文，按实际的摘要、出版社预览、作者存储库/源码范围记录，未声称已阅读全文，也未把其结果数字写入本稿实验结果。

## 作者与投稿前确认

作者顺序为Ziyuan Zhu、Sihao Liu、Haoyang Yu，Ziyuan Zhu兼通讯作者，ORCID0009-0005-1153-4926已加入且公开记录姓名相符；单位统一按用户确认。英文学院名沿用该院近年英文发表署名，邮编400016来自学院官网页脚；详细缙云街道地址与401331存在校方来源差异，未自行拼接详细通信地址。

无外部资助已确认。利益冲突声明、全体作者最终批准、排他投稿事实仍需作者确认。伦理依据使用2023年办法第32条第（一）项，并未捏造CQMU审批/豁免号。请核实公开数据免审条款前提及本机构实际要求。出处和中文解释已附。

文风已按英文anti-defensive-writing技能围绕可证实的机制贡献整理；期刊要求的AI辅助披露保留于Methods和Acknowledgments。无法通过改写承诺任何AI检测器结论。

## 复现

Python环境需要NumPy、pandas、Matplotlib；文稿渲染另需python-docx、Pandoc、XeLaTeX。

```bash
python research/analyze_main.py
python figures/plot_figures.py
python assemble_manuscript.py
python make_references.py
python build_documents.py
```

这些命令只做已归档数据的分析、绘图和排版，不训练模型、不申请GPU、不恢复停止实验。原始EEG保持在原发布库，源模型重训需另取原数据和科研代码。
