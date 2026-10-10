# 来源与复用说明

本文件包是 2026 年 10 月 11 日为求职作品集新整理的演示实现。

## 文件来源

- `engineering_surrogate/`：为当前演示新编写的 NumPy / Python 实现，使用 AI 辅助。
- `tests/`：针对当前演示分组、模型、校准和输入接口编写。
- `artifacts/`：由当前演示程序使用固定种子生成的合成数据、模型、表格和图，不含正式工程数据。
- 文档：根据本人已说明的经历和代码实际行为组织，不证明额外项目、岗位或发表状态。

本项目没有引入第三方源码包副本；运行所需 NumPy / Matplotlib 按各自许可证使用。方法来源与代码版权是不同事项。

## 方法参考

分层共形校准参考 Yonghoon Lee、Rina Foygel Barber、Rebecca Willett 的 [Distribution-free inference with hierarchical data](https://arxiv.org/abs/2306.06342)。本项目使用标准多项式扩展和 Ridge 最小二乘，不宣称新的数学方法。

## 公开与许可状态

当前交付为本地文件包，未建立远程仓库。未为权利人选定统一开源许可证。若选择公开为开源软件，再由权利人决定许可并添加对应 LICENSE；公开可见与授予复用权利不是同一事项。可参考 [GitHub 许可证说明](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)。

包内不包含正式论文、项目报告、原始实验资料、有限元输入或数据库、受保护模型权重、个人联系方式、本机私有路径、密钥或第三方参考实现副本。
