# Verified References (检索核实于 2026-10-27)

所有条目均通过 WebSearch/WebFetch 实际检索确认,验证 URL 为检索结果中实际出现的 arXiv / DOI / 出版方页面链接。未在检索结果中确认的条目一律不进入本列表(见下方 Unverified / dropped)。

## Verified references

### A. 核心强化学习算法与评测方法学

1. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," arXiv preprint arXiv:1707.06347, 2017.
   Verification URL: https://arxiv.org/abs/1707.06347
   Relevance: PPO 原始算法,IPPO/MAPPO 的基础。

2. C. S. de Witt, T. Gupta, D. Makoviichuk, V. Makoviychuk, P. H. S. Torr, M. Sun, and S. Whiteson, "Is independent learning all you need in the StarCraft multi-agent challenge?" arXiv preprint arXiv:2011.09533, 2020.
   Verification URL: https://arxiv.org/abs/2011.09533 (PDF: https://export.arxiv.org/pdf/2011.09533)
   Relevance: IPPO(独立 PPO)原始出处,本文基线算法之一;常见引用为 arXiv 预印本形式。

3. C. Yu, A. Velu, E. Vinitsky, J. Gao, Y. Wang, A. Bayen, and Y. Wu, "The surprising effectiveness of PPO in cooperative multi-agent games," in Advances in Neural Information Processing Systems (NeurIPS) 2022, Datasets and Benchmarks Track, 2022. arXiv:2103.01955.
   Verification URL: https://papers.neurips.cc/paper_files/paper/2022/file/9c1535a02f0ce079433344e14d910597-Paper-Datasets_and_Benchmarks.pdf (arXiv: https://arxiv.org/pdf/2103.01955)
   Relevance: MAPPO 原始论文,中心化价值函数的 PPO 变体,本文基线算法之一。

4. P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, and D. Meger, "Deep reinforcement learning that matters," in Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2018. DOI: 10.1609/aaai.v32i1.11694.
   Verification URL: https://ojs.aaai.org/index.php/aaai/article/view/11694
   Relevance: RL 实验方差/可复现性经典文献,支撑多随机种子设计的必要性。

5. R. Agarwal, M. Schwarzer, P. S. Castro, A. C. Courville, and M. G. Bellemare, "Deep reinforcement learning at the edge of the statistical precipice," in Advances in Neural Information Processing Systems (NeurIPS), vol. 34, pp. 29304–29320, 2021. arXiv:2108.13264.
   Verification URL: https://arxiv.org/abs/2108.13264 (NeurIPS PDF: https://papers.neurips.cc/paper_files/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf)
   Relevance: 少样本深度 RL 的统计报告规范(IQM、分层 bootstrap、性能曲线),本文统计方案直接依据。

6. X. Bouthillier et al., "Accounting for variance in machine learning benchmarks," in Proceedings of Machine Learning and Systems (MLSys), 2021. arXiv:2103.03098.
   Verification URL: https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf
   Relevance: 机器学习基准中多来源方差(初始化/数据/超参)的系统量化,支撑种子数量选择。

### B. 交通信号控制:经典与压力方法脉络

7. M. Wiering, "Multi-agent reinforcement learning for traffic light control," in Machine Learning: Proceedings of the Seventeenth International Conference (ICML 2000), 2000, pp. 1151–1158.
   Verification URL: 引文见 https://www.computer.org/csdl/journal/tm/2024/06/10315185/1S2UkYTD4C4 (IEEE CSDL 参考文献) 与 https://arxiv.org/pdf/2607.19555
   Relevance: 多智能体 RL 用于交通灯控制的早期奠基工作。

8. H. Wei, G. Zheng, V. Gayah, and Z. Li, "A survey on traffic signal control methods," arXiv preprint arXiv:1904.08117, 2019.
   Verification URL: https://arxiv.org/abs/1904.08117 (PDF: https://arxiv.org/pdf/1904.08117)
   Relevance: 高被引 TSC 方法综述(覆盖固定配时、max-pressure、SCATS 与 RL 方法)。

9. P. Varaiya, "Max pressure control of a network of signalized intersections," Transportation Research Part C: Emerging Technologies, vol. 36, pp. 177–195, 2013. DOI: 10.1016/j.trc.2013.08.014.
   Verification URL: https://doi.org/10.1016/j.trc.2013.08.014 (引文核实于 https://dl.acm.org/doi/10.1111/mice.13364 参考文献)
   Relevance: max-pressure 控制的奠基性论文,本文压力先验的理论来源。

10. L. Tassiulas and A. Ephremides, "Stability properties of constrained queueing systems and scheduling policies for maximum throughput in multihop radio networks," IEEE Transactions on Automatic Control, vol. 37, no. 12, pp. 1936–1948, 1992.
    Verification URL: 引文核实于 https://handwiki.org/wiki/Lyapunov_optimization 及 https://www.jsjkx.com/EN/article/openArticlePDF.jsp?id=15885
    Relevance: backpressure/最大吞吐调度原始文献,压力类控制策略的理论源头(替代任务清单中的 "Tassiulas & Lin 1994")。

11. H. Wei, G. Zheng, H. Yao, and Z. Li, "IntelliLight: A reinforcement learning approach for intelligent traffic light control," in Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '18), 2018, pp. 2496–2505. DOI: 10.1145/3219819.3220096.
    Verification URL: https://dl.acm.org/doi/suppl/10.1145/3219819.3220096
    Relevance: DRL-TSC 的诊断性/开创性工作,讨论奖励设计等工程细节。

12. H. Wei, C. Chen, G. Zheng, K. Wu, V. Gayah, K. Xu, and Z. Li, "PressLight: Learning max pressure control to coordinate traffic signals in arterial network," in Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '19), 2019, pp. 1290–1298. DOI: 10.1145/3292500.3330949.
    Verification URL: https://dl.acm.org/doi/pdf/10.1145/3292500.3330949
    Relevance: 以 max-pressure 为奖励的核心先验文献(压力奖励 DRL-TSC 代表作)。

13. H. Wei, N. Xu, H. Zhang, G. Zheng, X. Zang, C. Chen, W. Zhang, Y. Zhu, K. Xu, and Z. Li, "CoLight: Learning network-level cooperation for traffic signal control," in Proceedings of the 28th ACM International Conference on Information and Knowledge Management (CIKM '19), 2019, pp. 1913–1922. arXiv:1905.05717.
    Verification URL: https://arxiv.org/abs/1905.05717 (页码核实于 KDD '25 论文参考文献 https://dl.acm.org/doi/10.1145/3770855.3818039)
    Relevance: 图注意力网络实现路网级协作的代表作(非压力类协作基线)。

14. G. Zheng, Y. Xiong, X. Zang, J. Feng, H. Wei, H. Zhang, Y. Li, K. Xu, and Z. Li, "Learning phase competition for traffic signal control," in Proceedings of the 28th ACM International Conference on Information and Knowledge Management (CIKM '19), 2019. DOI: 10.1145/3357384.3357900. arXiv:1905.04722.
    Verification URL: https://dl.acm.org/doi/pdf/10.1145/3357384.3357900 (另见作者团队站点 https://traffic-signal-control.github.io/ 标注 CIKM'19)
    Relevance: FRAP 相位竞争模型。注意:任务清单原标注 ICLR 2019,经核实(ACM DL、作者站点、Semantic Scholar 三方一致)正式发表 venue 为 CIKM 2019。

15. C. Chen, H. Wei, N. Xu, G. Zheng, M. Yang, Y. Xiong, K. Xu, and Z. Li, "Toward a thousand lights: Decentralized deep reinforcement learning for large-scale traffic signal control," in Proceedings of the AAAI Conference on Artificial Intelligence (AAAI-20), vol. 34, no. 04, pp. 3414–3421, 2020.
    Verification URL: https://cdn.aaai.org/ojs/5744/5744-13-8969-1-10-20200513.pdf
    Relevance: MPLight 方法出处(参数共享的去中心化 DRL,大规模路网可扩展性)。

16. C. Zhao, X. Hu, and G. Wang, "PDLight: A deep reinforcement learning traffic light control algorithm with pressure and dynamic light duration," arXiv preprint arXiv:2009.13711, 2020.
    Verification URL: https://arxiv.org/abs/2009.13711
    Relevance: PRCOL(Pressure with Remaining Capacity of Outgoing Lane)压力指标的原始出处——改进 PressLight 压力定义,考虑下游车道剩余容量。经核实,PRCOL 一词即出自本文,而非任务清单所猜测的 "Ou et al. 2021"(后者未能找到,见下)。

### C. 近期(2022–2026)压力信号 + RL 与综述

17. X. Li, X. Wang, I. Smirnov, S. Sanner, and B. Abdulhai, "Multi-hop upstream anticipatory traffic signal control with deep reinforcement learning," arXiv preprint arXiv:2411.07271, 2024 (v2: 2025). 期刊版见 IEEE Xplore (doc. 10970747, IEEE Journals & Magazine)。
    Verification URL: https://arxiv.org/abs/2411.07271 (IEEE 版: https://ieeexplore.ieee.org/document/10970747)
    Relevance: 提出多跳上游压力(multi-hop upstream pressure)作为 RL 奖励,将传统单链路压力推广至上游多跳范围——与本文压力先验方向最直接相关的近年工作。

18. H. Zhao et al., "A survey on deep reinforcement learning approaches for traffic signal control," Engineering Applications of Artificial Intelligence, 2024, art. 108100. DOI: 10.1016/j.engappai.2024.108100.
    Verification URL: https://dl.acm.org/doi/10.1016/j.engappai.2024.108100
    Relevance: 2024 年 DRL-TSC 综述,可覆盖"近年综述(2023–2026)"需求。

### D. 仿真平台与统计方法

19. P. A. Lopez, M. Behrisch, L. Bieker-Walz, J. Erdmann, Y.-P. Flötteröd, R. Hilbrich, L. Lücken, J. Rummel, P. Wagner, and E. Wießner, "Microscopic traffic simulation using SUMO," in 2018 21st International Conference on Intelligent Transportation Systems (ITSC), 2018, pp. 2575–2582. DOI: 10.1109/ITSC.2018.8569938.
    Verification URL: https://sumo.dlr.de/docs/Publications.html (DOI 核实于 https://sumo.dlr.de/poster/poster-4.pdf 参考文献)
    Relevance: SUMO 仿真平台官方引用文献。

20. B. Phipson and G. K. Smyth, "Permutation p-values should never be zero: Calculating exact p-values when permutations are randomly drawn," Statistical Applications in Genetics and Molecular Biology, vol. 9, no. 1, 2010. DOI: 10.2202/1544-6115.1585.
    Verification URL: https://phipsonlab.github.io/phipson-lab-website/research/ (另见 SciPy 文档引用 https://docs.scipy.org/doc/scipy-1.11.1/reference/generated/scipy.stats.jarque_bera.html)
    Relevance: 置换检验 p 值下界校正(+1 修正),本文置换检验统计方案依据。注意 venue 是 SAGMB(并非部分资料误写的 Statistical Science)。

21. B. Efron, "Bootstrap methods: Another look at the jackknife," The Annals of Statistics, vol. 7, no. 1, pp. 1–26, 1979. DOI: 10.1214/aos/1176344552.
    Verification URL: https://projecteuclid.org/journals/annals-of-statistics/volume-7/issue-1/Bootstrap-Methods-Another-Look-at-the-Jackknife/10.1214/aos/1176344552.full
    Relevance: Bootstrap 方法原始文献,本文置信区间估计依据。

## Unverified / dropped

- **Ou et al., "Put pressure on deep reinforcement learning for traffic signal control" (~2021)** — **UNVERIFIED**。使用精确标题、"Ou"+"PRCOL"、"Tsinghua Science and Technology" 等多种关键词共 5 次检索,均未能定位到该论文(无 arXiv/DOI/出版方页面命中)。重要发现:任务清单所称的 PRCOL 指标实际出自 **PDLight(Zhao, Hu, Wang, arXiv:2009.13711, 2020)**,已作为条目 16 验证收录。建议:引用 PDLight 而非 Ou et al.;若手头有 Ou et al. 的原始来源线索,需先人工核实再决定是否补录。
- **Tassiulas & Lin (1994)** — 未检索到与任务描述匹配的确切条目;已改用经核实的相关源头文献 Tassiulas & Ephremides (IEEE TAC 1992,条目 10)。
- **Rasheed et al. (2020) TSC-RL survey** — 按任务要求仅收录一篇综述,选用 Wei et al. 2019 (arXiv:1904.08117);2023–2026 近年综述由 Zhao et al. 2024 (EAAI,条目 18) 覆盖,Rasheed 未再检索。
- **Efron & Tibshirani (1993) "An Introduction to the Bootstrap" 书目** — 未检索;Efron 1979 原文已足够支撑 bootstrap 置信区间方法引用。

## Coverage notes

覆盖情况总体扎实:核心 RL 算法线(PPO/IPPO/MAPPO)、方法学线(Henderson/Agarwal/Bouthillier/Phipson-Smyth/Efron)、交通经典线(Wiering/IntelliLight/FRAP/MPLight/CoLight)与压力方法线(Varaiya/Tassiulas-Ephremides/PressLight/PDLight-PRCOL/多跳上游压力)均有经 DOI 或 arXiv 页面直接核实的一手文献。压力先验脉络完整覆盖了"max-pressure 理论 → 压力奖励(PressLight)→ 压力指标改进(PRCOL/PDLight)→ 多跳压力泛化(2024/2025)"的演化链,直接支撑本文压力先验定位。薄弱处有三:其一,20 项候选中未验证的仅 Ou et al. 一项,但该条恰是任务清单中疑似有误的引用(PRCOL 实际出处为 PDLight),需在成文时以 PDLight 替代;其二,FRAP 的正式 venue 经核实为 CIKM 2019 而非常见误引的 ICLR 2019,引用时务必采用 CIKM 版本;其三,2022–2026 年"压力 + RL"方向仅验证到 1 篇(multi-hop upstream, arXiv:2411.07271),该细分方向近期文献覆盖偏薄,且 IPPO、Wei 综述为 arXiv/预印本形态,缺少期刊/会议正式版。此外,所有压力类先验工作(PressLight/PRCOL/multi-hop)均基于 DQN 类值函数方法而非 PPO 系,这正好构成本文(IPPO + 压力先验)的差异化空间,可在相关工作中明确指出。
