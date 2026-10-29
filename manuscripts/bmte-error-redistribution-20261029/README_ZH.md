# BMTE英文实证稿与复算材料

论文题目：**Error redistribution across conventional and neural motor imagery EEG recalibration**。

研究问题统一为：解码器更新如何通过纠错、引错和参与者差异形成校准净收益。正文用传统管线定位更新组件，用神经配置比较解释相同净准确率背后的预测变化。贡献落在有对照的系统实证证据上，既有算法、预测变化概念和分解定义保留出处。

主分析为五队列56名参与者、212个目标会话。Kumar开发6人、Dreyer源模型更替85人分别报告。预算保留有效标签与采集位置的区别；各队列保留原模型、预处理和参考路径。全部配置和预算在补充数据中保留。新稿没有新增训练、模型推理或p值检验。

## 交付文件

- `Manuscript.docx`、`Manuscript.pdf`：英文主稿。
- `Supplementary_Materials.docx`、`Supplementary_Materials.pdf`：完整补充方法、结果和4张补图。
- `submission/Cover_Letter.docx`：投稿信，含相关公开稿件和数据关系的编辑说明。
- `submission/Ethical_and_Legal_Declarations.docx`：按期刊官方模板填好的独立声明。
- `submission/Figure_captions.docx`、`submission/Title_page.docx`：独立图注与标题页。
- `submission/Submission_Fields.txt`：可复制的新标题、短标题、摘要和关键词。
- `figures/Fig1–Fig4`：4张主图，各有TIFF/PNG/PDF/SVG。
- `submission/Supplementary_Data_and_Code.zip`：期刊用的小型复算数据与代码包，各补充文件小于10MB。

旧稿与原始结果另存。当前目录是重写后的作者审阅版本，尚未实际投稿；投稿入口为 https://mc.manuscriptcentral.com/bmt 。使用本次新稿时，需要同步替换系统里此前填入的标题、摘要、关键词和已上传的文件。作者应阅读全文，核对贡献和编辑声明后再完成系统提交。

## 作者与声明

Ziyuan Zhu、Sihao Liu、Haoyang Yu及邮箱、ORCID沿用已确认资料。三位作者单位均为重庆医科大学人工智能医学学院缙云校区，61 Daxuecheng Middle Road, Shapingba District, Chongqing401331, China。无利益冲突、无外部资助按用户确认填写。公开数据二次分析采用国家办法第32条第（一）项免伦理审查依据，原采集信息逐数据集引用。投稿表不虚构机构批准号或未获证明的原始同意信息。

## 复算与排版

```bash
python analysis/analyze_integrated.py
python analysis/check_analysis.py
python figures/plot_integrated.py
python build_support_materials.py
python assemble_manuscript.py
python build_supplement.py
python build_documents.py
python make_submission_bundle.py
python validate_package.py
python package_delivery.py
```

数据分析和绘图脚本读取本地归档输入。文档排版另需Pandoc、XeLaTeX和python-docx；依赖版本见 `requirements-analysis.txt`。输入哈希、配对检查和独立数值复核均随包保存。完整作者工作ZIP用于保管；在线补充提交使用专门的小型数据代码ZIP。
