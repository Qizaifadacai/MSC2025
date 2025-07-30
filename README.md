# Live/Dead Cell Analysis Tool

## 功能描述
这是一个用于分析荧光显微镜下活/死细胞图像的自动化工具。主要功能包括：

1. **双通道图像处理**
   - 绿色通道 (GFP): 识别活细胞
   - 蓝色通道 (DAPI): 识别死细胞
   - 支持16位 TIFF 格式图像

2. **自动细胞计数**
   - 使用改进的分水岭算法分离重叠细胞
   - 智能阈值处理，适应不同亮度条件
   - 基于形态学和面积的细胞筛选

3. **结果可视化**
   - 生成带比例尺的合成图像
   - 细胞边界标注
   - 统计数据可视化（饼图等）
   - 导出详细的分析报告

4. **批量处理**
   - 支持多文件夹批处理
   - 自动生成汇总报告
   - CSV格式导出统计结果

## 使用方法
1. 安装依赖：
```bash
pip install numpy pillow opencv-python scipy scikit-image matplotlib
```

2. 运行脚本：
```python
python LiveDead_Cell_counting.py
```

## 输出结果
- 合成的RGB图像
- 细胞分析图（带标注）
- 统计报告（CSV和TXT格式）
- 分析结果可视化图表

## 依赖库
- numpy
- PIL (Pillow)
- opencv-python
- scipy
- scikit-image
- matplotlib
