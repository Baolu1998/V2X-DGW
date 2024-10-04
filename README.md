# V2X-DGW

Data and code are coming soon.

abstract:
Current LiDAR-based Vehicle-to-Everything (V2X) multi-agent  perception systems have shown the significant success on 3D object detection. While these models perform well in the trained clean weather, they struggle in unseen adverse weather conditions with the domain gap. In this paper,  we propose a Domain Generalization based approach, named V2X-DGW, for LiDAR-based 3D object detection on multi-agent perception system under adverse weather conditions. Our research aims to not only maintain favorable multi-agent performance in the clean weather but also promote the performance in the unseen adverse weather conditions by learning only on the clean weather data. To realize the Domain Generalization, we first introduce the Adaptive Weather Augmentation (AWA) to mimic the unseen adverse weather conditions, and then propose two alignments for generalizable representation learning: Trust-region Weather-invariant Alignment (TWA) and Agent-aware Contrastive  Alignment (ACA). To evaluate this research, we add Fog, Rain, Snow conditions on two publicized multi-agent datasets based on physics-based models, resulting in two new datasets: OPV2V-w and V2XSet-w. Extensive experiments demonstrate that our V2X-DGW achieved significant improvements in the unseen adverse weathers.



@article{li2024v2x,
  title={V2X-DGW: Domain Generalization for Multi-agent Perception under Adverse Weather Conditions},
  author={Li, Baolu and Li, Jinlong and Liu, Xinyu and Xu, Runsheng and Tu, Zhengzhong and Guo, Jiacheng and Li, Xiaopeng and Yu, Hongkai},
  journal={arXiv preprint arXiv:2403.11371},
  year={2024}
}
