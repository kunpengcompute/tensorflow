<div align="center">

**鲲鹏Tensorflow扩展**
===============================

</div>

鲲鹏Tensorflow是基于开源Tensorflow的高性能推理加速扩展，为AI应用提供基于鲲鹏CPU极致性能。

## 架构
框架层：调度器优化。

Kernel层：自定义算子，基于KDNN提供鲲鹏高性能DNN算子。

grappler层：基于ANNC提供鲲鹏图编译器。

## 安装

### 硬件要求

鲲鹏Tensorflow扩展提供对鲲鹏CPU的支持。

### 软件依赖

鲲鹏DNN库：KDNN。

鲲鹏AI编译器：ANNC。

### 安装方式:

* 源码构建: 待补充。


### 支持的版本

| Kunpeng Tensorflow  | Stock TensorFlow |
| -------             | -----------      |
| v2.15.0             | 2.15             |
| v2.2.0              | 2.2              |


## 特性说明

### 线程调度优化特性

#### 简介

为提升TensorFlow Serving（以下简称TF Serving）推理性能，鲲鹏BoostKit提出了TensorFlow Serving线程调度优化方案。传统TensorFlow使用算子间的线程池并行计算不同的算子，虽可实现没有数据依赖的算子的并发执行，但在高并发场景下，多Session共享算子间线程池会导致任务抢占，严重降低整图计算效率。针对这一痛点，鲲鹏BoostKit改进了算子调度算法，并加入了其他线程管理优化，有效提升了高并发场景下的模型推理吞吐量。

#### 详细说明
参考：[TensorFlow Serving线程调度优化](https://www.hikunpeng.com/document/detail/zh/SRA/accelFeatures/TFTSO/kunpengsra_tfserving_20_0002.html)

#### 版本兼容性
v2.15

## 支持
如果使用过程中有任何问题，或者需要反馈特性需求和bug报告，可以提交[isssues](https://gitcode.com/BoostKit/tensorflow/issues)联系我们。

## 许可协议
[Apache License 2.0](LICENSE)
通过下载并使用此源码及其附带的软件，您即同意遵守软件许可协议中的条款和条件。
